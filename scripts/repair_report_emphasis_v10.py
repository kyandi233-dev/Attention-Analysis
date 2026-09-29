"""Apply a restrained bold/underline hierarchy to the v9 story paragraphs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree


EMPHASIS = {
    17: {
        "bold": ["FocusWave无接触多模态持续注意智能测评系统"],
        "underline": ["标准化任务、多源同步采集、动态特征分析与跨个体预测"],
    },
    18: {
        "bold": ["61名参与者共完成116轮持续注意测评"],
        "underline": ["行为表现、主观注意与生理状态能够在相同任务阶段和近期时间窗口内进行分析"],
    },
    19: {
        "bold": ["反应时变异显著增加", "眨眼频率高于个人平均水平时，参与者更可能报告任务无关思维或思维空白，并出现较多遗漏"],
        "underline": ["近红外瞳孔、身体动作和毫米波心肺记录从不同侧面补充了任务行为与即时报告"],
    },
    20: {
        "bold": ["任务行为对任务聚焦报告具有较稳定的区分能力", "传感信息在行为基础上的改善尚不稳定"],
        "underline": ["仅传感信息联合模型、行为基础上的成对增量和十三项完整组合的逐项移除"],
    },
    21: {
        "bold": ["FocusWave已形成从任务设计、无接触多源采集到动态特征分析与跨个体预测的测评流程"],
        "underline": ["完成面向使用者的界面与交互原型"],
    },
    27: {
        "bold": ["非接触感知为降低测评中的主动操作负担提供了技术条件"],
        "underline": ["将信号质量、指标含义与预测评价结合起来，有助于识别在实际测量条件下具有稳定价值的信息"],
    },
    29: {
        "bold": ["构建并评价低干扰、非接触的多模态持续注意测评系统"],
        "underline": ["比较传感信息的独立表现及其在行为基础上的增量", "跨参与者验证、概率校准和设备配置评价"],
    },
    35: {
        "bold": ["持续注意是指个体在一段时间内持续将注意投入当前任务"],
        "underline": ["两类任务都涉及注意的持续维持，但其中包含的感知、规则保持和反应控制过程并不完全相同"],
    },
    48: {
        "bold": ["持续性注意反应任务（Sustained Attention to Response Task，SART）"],
        "underline": ["SART 中的表现与日常生活中的注意失误有关"],
    },
    49: {
        "bold": ["连续的逐试次行为记录"],
        "underline": ["反应速度、错误和反应稳定性如何随任务进行发生变化"],
    },
    55: {
        "bold": ["SART 表现需要结合参与者采用的反应策略进行理解"],
        "underline": ["SART 中的反应速度和准确率常常相互影响"],
    },
    56: {
        "bold": ["思维探针可以补充这一部分信息"],
        "underline": ["探针捕获和自我捕获", "即时探针减少了回忆造成的偏差，也更容易与探针前的行为和生理信号进行时间对应"],
    },
    58: {
        "bold": ["思维探针具有理论和经验依据，适合作为主观状态参照"],
        "underline": ["题干、选项设置和报告要求都会影响测量结果"],
    },
    63: {
        "bold": ["瞳孔大小由交感神经和副交感神经共同调节"],
        "underline": ["除了光照、注视位置等视觉因素外，心理努力、认知控制和任务事件也会引起瞳孔变化"],
    },
    72: {
        "bold": ["眨眼与认知状态的关系随任务要求而变化"],
        "underline": ["视觉加工、认知负荷和困倦程度"],
    },
    78: {
        "bold": ["同一种动作在不同任务中的意义可能完全不同"],
        "underline": ["姿态和头部活动的含义还会受到具体场景影响"],
    },
    80: {
        "bold": ["人工识别的小动作与视频算法提取的整体动作量各有侧重"],
        "underline": ["明确指标的定义，并控制非行为性的图像变化"],
    },
    97: {
        "bold": ["需要分别检验传感信息独立运行的能力，以及它对任务行为的补充价值"],
        "underline": ["一个变量可以单独预测主观状态，同时与已有信息高度重叠"],
    },
    104: {
        "bold": ["各类观测如何反映注意变化、传感信息能否补充行为预测，以及模型面对新参与者时的表现"],
        "underline": ["概率可靠性、有效覆盖和设备配置"],
    },
    245: {
        "bold": ["设备组合模型均保留行为参照"],
        "underline": ["仅依赖传感信息的使用方式则由 4.6.5 的传感联合模型单独评价"],
    },
    247: {
        "bold": ["设备配置根据实际纳入的有效指标建立模型"],
        "underline": ["配置中的每类传感设备均应提供相应特征"],
    },
    474: {
        "bold": ["任务行为与即时报告共同呈现注意状态的变化", "行为信息的稳定价值"],
        "underline": ["传感信息自身的表现、行为基础上的增量和完整组合中的条件价值", "概率校准与设备配置结果"],
    },
    494: {
        "bold": ["后续研究将围绕独立参与者、目标场景和个体校准展开"],
        "underline": ["增加不同人群及少数注意类别的观测", "在自然学习与工作任务中验证"],
    },
    591: {
        "bold": ["事后问卷与任务内报告的场次级一致性分析"],
        "underline": [],
    },
    613: {
        "bold": ["整场自评走神比例较高的场次，任务内非专注探针比例也较高"],
        "underline": ["最高走神等级仅包含 9 个场次"],
    },
    615: {
        "bold": [],
        "underline": ["毫米波心率和呼吸率与整场走神等级的关联区间包含无关联值"],
    },
    713: {
        "bold": ["参与者等权宏平均对数损失"],
        "underline": ["直接比较采用共同分析集合"],
    },
    718: {
        "bold": ["受试者工作特征曲线下面积（area under the receiver operating characteristic curve [AUROC]）为参与者宏平均"],
        "underline": ["校准回归按探针合并"],
    },
}


def span_mask(text: str, phrases: list[str]):
    mask = [False] * len(text)
    for phrase in phrases:
        assert text.count(phrase) == 1, (phrase, text.count(phrase))
        start = text.index(phrase)
        mask[start:start + len(phrase)] = [True] * len(phrase)
    return mask


def has_tag(rpr, name: str):
    if rpr is None:
        return False
    found = rpr.xpath(f'./w:{name}')
    return bool(found) and found[0].get(qn('w:val'), '1') not in {'0', 'false', 'none'}


def adjusted_properties(source, bold: bool, underline: bool, mark: bool):
    rpr = copy.deepcopy(source._r.rPr) if source._r.rPr is not None else OxmlElement('w:rPr')
    for name in ('b', 'bCs', 'u'):
        for node in list(rpr.xpath(f'./w:{name}')):
            rpr.remove(node)
    if bold:
        rpr.append(OxmlElement('w:b'))
    if underline:
        node = OxmlElement('w:u')
        node.set(qn('w:val'), 'single')
        rpr.append(node)
    if mark:
        for node in list(rpr.xpath('./w:highlight')):
            rpr.remove(node)
        node = OxmlElement('w:highlight')
        node.set(qn('w:val'), 'yellow')
        rpr.append(node)
    return rpr


def emit_run(rpr, value):
    run = OxmlElement('w:r')
    run.append(copy.deepcopy(rpr))
    child = OxmlElement('w:t')
    if value[:1].isspace() or value[-1:].isspace():
        child.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    child.text = value
    run.append(child)
    return run


def restyle(paragraph, plan):
    old = paragraph.text
    bold_mask = span_mask(old, plan['bold'])
    under_mask = span_mask(old, plan['underline'])
    assert not any(a and b for a, b in zip(bold_mask, under_mask))
    original_runs = list(paragraph.runs)
    assert original_runs and all(child.tag in {qn('w:pPr'), qn('w:r')} for child in paragraph._p)
    pieces = []
    position = 0
    for source in original_runs:
        original_bold = has_tag(source._r.rPr, 'b')
        original_under = has_tag(source._r.rPr, 'u')
        for char in source.text:
            new_bold, new_under = bold_mask[position], under_mask[position]
            changed = original_bold != new_bold or original_under != new_under
            properties = adjusted_properties(source, new_bold, new_under, changed)
            key = etree.tostring(properties)
            if pieces and pieces[-1][0] == key:
                pieces[-1] = (key, pieces[-1][1] + char, properties)
            else:
                pieces.append((key, char, properties))
            position += 1
    assert position == len(old)
    for source in original_runs:
        paragraph._p.remove(source._r)
    for _, value, properties in pieces:
        paragraph._p.append(emit_run(properties, value))
    assert paragraph.text == old
    return {'bold': plan['bold'], 'underline': plan['underline']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    document = Document(args.source)
    record = {}
    for index, plan in EMPHASIS.items():
        paragraph = document.paragraphs[index]
        before = etree.tostring(paragraph._p.pPr)
        record[str(index)] = restyle(paragraph, plan)
        assert etree.tostring(paragraph._p.pPr) == before
    xml = etree.tostring(document._element, xml_declaration=True, encoding='UTF-8', standalone=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source) as source_zip, zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as output_zip:
        for info in source_zip.infolist():
            output_zip.writestr(info, xml if info.filename == 'word/document.xml' else source_zip.read(info.filename))
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    report = {'source': str(args.source), 'source_sha256': digest(args.source), 'output': str(args.output), 'output_sha256': digest(args.output), 'restyled_paragraphs': record, 'text_changes': 0, 'figure_changes': 0, 'model_refits': 0}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output_sha256': report['output_sha256'], 'restyled_paragraphs': list(record)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
