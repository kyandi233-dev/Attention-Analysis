"""Revise five report bridge paragraphs while preserving all Word structure."""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree


REVISIONS = {
    "结果呈现了持续注意在不同时间尺度上的变化。随着任务持续，反应时变异显著增加，较低警觉状态伴随更多No-Go误按和筛选后遗漏。眨眼频率高于个人平均水平时，参与者更可能报告任务无关思维或思维空白，并出现较多遗漏。近红外瞳孔测量保留了水平、波动和变化趋势等信息，毫米波则拓展了心率与呼吸活动的无接触记录方式。各类指标分别描述任务执行、意识内容、眼部活动和自主生理状态，使系统能够从多个角度观察注意变化，并为后续选择有价值的测评指标提供依据。":
    "结果呈现了持续注意在任务进程与探针前近期行为中的变化。随着任务持续，反应时变异显著增加；较低警觉状态伴随更多No-Go误按和筛选后遗漏。眨眼频率高于个人平均水平时，参与者更可能报告任务无关思维或思维空白，并出现较多遗漏。近红外瞳孔、身体动作和毫米波心肺记录从不同侧面补充了任务行为与即时报告，使各类观测能够沿同一任务时间轴比较。",
    "在智能评估方面，FocusWave建立了行为、单模态与多模态预测模型，采用留一参与者交叉验证评价模型对新使用者的表现。行为信息提供了较稳定的任务聚焦报告区分能力，眼部信息也呈现一定的独立排序能力。多模态比较进一步明确了不同信息在独立预测、补充行为和完整组合中的作用。现阶段，传感信息在行为基础上的预测改善仍有待进一步验证，相关结果为优化特征组合和概率估计提供了具体方向。":
    "跨参与者预测表明，任务行为对任务聚焦报告具有较稳定的区分能力，眼部信息也呈现一定的独立排序能力。仅传感信息联合模型、行为基础上的成对增量和十三项完整组合的逐项移除，分别评价了传感信息自身、补充行为及进入完整组合后的预测价值。传感信息在行为基础上的改善尚不稳定。模型评价同时考察预测损失、类别区分与概率校准，设备配置分析进一步呈现不同输入条件下的有效覆盖和折外表现。",
    "本研究据此以构建并评价低干扰、非接触的多模态持续注意测评系统为目标，在统一任务中建立行为与即时主观报告参照，同步采集眼部、身体动作及心肺相关信号，评价各类信息的测量质量、状态关联与跨参与者预测表现。本研究的验证以探针前时间窗口和随后报告为对应单位，重点考察传感信息自身的预测能力及其在行为基础上的增量。系统的后续应用方向是在完成目标场景验证后，以近期传感信息估计使用者报告任务聚焦或其他注意内容的概率，逐步减少常规使用中对专门实验任务和频繁询问的依赖，使持续注意测评更自然地融入日常活动。":
    "本研究据此以构建并评价低干扰、非接触的多模态持续注意测评系统为目标。在统一任务中，行为与即时主观报告构成状态参照，眼部、身体动作及心肺相关信号提供同步观测。研究考察各类指标的有效覆盖、任务进程变化和主观状态关联，并以探针前时间窗口预测随后报告，比较传感信息的独立表现及其在行为基础上的增量。跨参与者验证、概率校准和设备配置评价共同界定系统在当前任务中的测评表现与使用条件。后续面向目标场景的验证将进一步检验减少专门任务和频繁询问的使用方式。",
    "前述理论与技术研究为非接触持续注意测评提供了任务参照和观测方法。FocusWave在此基础上，探索如何将连续采集的多源信号转化为心理含义明确、能够面向新参与者评价的状态估计。研究围绕受控任务中的系统构建与验证，依次完成同步采集、指标形成、主观报告预测和设备配置评价，使测量依据与系统应用相衔接。":
    "前述理论与技术研究为非接触持续注意测评提供了任务参照和观测方法。FocusWave在统一任务时间轴上对应行为、即时报告与多源传感记录，考察各类观测如何反映注意变化、传感信息能否补充行为预测，以及模型面对新参与者时的表现。研究进一步结合概率可靠性、有效覆盖和设备配置，评价系统的使用条件。",
    "本研究在61名参与者的116场实验中，建立了涵盖任务设计、多源采集、指标分析和跨参与者预测的非接触持续注意测评流程。任务行为提供了可靠的比较基准，眨眼频率与部分注意内容及遗漏表现的关联，也为非接触观察注意变化提供了线索。多模态比较进一步明确了各类信息在现有测量条件下的作用。结合这些结果，后续工作可以更有针对性地提高传感质量、检验信息互补性，并完善面向新使用者的概率输出。":
    "本研究在61名参与者的116场实验中，建立了涵盖任务设计、多源采集、指标分析和跨参与者预测的非接触持续注意测评流程。任务行为与即时报告共同呈现注意状态的变化，眨眼频率与部分注意内容及遗漏表现相联系。跨参与者预测显示了行为信息的稳定价值，也区分了传感信息自身的表现、行为基础上的增量和完整组合中的条件价值。概率校准与设备配置结果进一步说明了系统在不同测量条件下的输出特点，为指标优化和后续应用评价提供了依据。",
}


def copy_text_run(source, value: str, changed: bool):
    run = OxmlElement("w:r")
    if source._r.rPr is not None:
        run.append(copy.deepcopy(source._r.rPr))
    if changed:
        rpr = run.rPr
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            run.insert(0, rpr)
        for previous in list(rpr.xpath("./w:highlight")):
            rpr.remove(previous)
        highlight = OxmlElement("w:highlight")
        highlight.set(qn("w:val"), "yellow")
        rpr.append(highlight)
    node = OxmlElement("w:t")
    if value[:1].isspace() or value[-1:].isspace():
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = value
    run.append(node)
    return run


def replace_preserving_runs(paragraph, replacement: str):
    old = paragraph.text
    assert old != replacement
    runs = list(paragraph.runs)
    assert runs and all(child.tag in {qn("w:pPr"), qn("w:r")} for child in paragraph._p)
    spans = []
    cursor = 0
    for run in runs:
        spans.append((cursor, cursor + len(run.text), run))
        cursor += len(run.text)
    assert cursor == len(old)
    base = next((run for run in runs if run.bold is not True), runs[0])
    emitted = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, old, replacement, autojunk=False).get_opcodes():
        if tag == "equal":
            for left, right, source in spans:
                start, end = max(a, left), min(b, right)
                if start < end:
                    emitted.append(copy_text_run(source, old[start:end], False))
        elif tag in {"replace", "insert"} and c < d:
            emitted.append(copy_text_run(base, replacement[c:d], True))
    for run in runs:
        paragraph._p.remove(run._r)
    for run in emitted:
        paragraph._p.append(run)
    assert paragraph.text == replacement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    document = Document(args.source)
    changed = []
    for before, after in REVISIONS.items():
        matches = [(i, p) for i, p in enumerate(document.paragraphs) if p.text == before]
        assert len(matches) == 1, (before[:40], len(matches))
        i, paragraph = matches[0]
        style_before = etree.tostring(paragraph._p.pPr)
        replace_preserving_runs(paragraph, after)
        assert etree.tostring(paragraph._p.pPr) == style_before
        changed.append({"paragraph_index": i, "before": before, "after": after})
    xml = etree.tostring(document._element, xml_declaration=True, encoding="UTF-8", standalone=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source) as source_zip, zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output_zip:
        for entry in source_zip.infolist():
            output_zip.writestr(entry, xml if entry.filename == "word/document.xml" else source_zip.read(entry.filename))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"source": str(args.source), "source_sha256": digest(args.source), "output": str(args.output), "output_sha256": digest(args.output), "changed_paragraphs": changed, "model_refits": 0, "figures_removed": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_sha256": digest(args.output), "paragraphs_changed": len(changed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
