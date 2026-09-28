"""Publish reviewed text and aggregate evidence into existing formal report entries."""
from pathlib import Path
import argparse,json,re,shutil
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--docx',type=Path,required=True);ap.add_argument('--formal-repo',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args();r=a.formal_repo;d=Document(a.docx)
    m=json.loads((a.results/'audit15_edit_manifest.json').read_text(encoding='utf-8'))
    # Apply exact paragraph replacements only where the current canonical text matches.
    for f in list((r/'国赛报告/章节草稿').glob('*.md'))+list((r/'国赛报告/附录').glob('*.md')):
        text=f.read_text(encoding='utf-8');old=text
        for change in m['changes']:
            before=change['before'];after=change['after']
            if isinstance(before,str) and len(before)>24 and change['location'].startswith('paragraph ') and before in text:text=text.replace(before,after)
        text=text.replace('线性概率模型','L2 正则化逻辑回归').replace('明确遗漏','无已检测时序歧义遗漏').replace('清洗后遗漏','无已检测时序歧义遗漏')
        if text!=old:f.write_text(text,encoding='utf-8')
    def mdtable(t):
        rows=['| '+' | '.join(c.text.replace('\n','<br>').replace('|','\\|') for c in row.cells)+' |' for row in t.rows]
        rows.insert(1,'| '+' | '.join('---' for _ in t.rows[0].cells)+' |');return '\n'.join(rows)
    def extract(start,end):
        active=False;out=[]
        for el in d._element.body:
            if el.tag==qn('w:p'):
                p=Paragraph(el,d);text=p.text
                if text.startswith(end):break
                if text.startswith(start):active=True
                if not active:continue
                bs=p._p.xpath('.//a:blip')
                if bs:
                    part=d.part.related_parts[bs[0].get(qn('r:embed'))];fn='20260928_review_'+Path(str(part.partname)).name
                    (r/'国赛报告/assets'/fn).write_bytes(part.blob);out.append(f'![结果图](../assets/{fn})')
                elif text:out.append(('# ' if text.startswith(start) else '## ' if re.match(r'^(\d\.\d\.\d|[A-G]\.\d)\s',text) else '')+text)
            elif el.tag==qn('w:tbl') and active:out.append(mdtable(Table(el,d)))
        return '\n\n'.join(out)+'\n'
    for start,end,filename in [('5.6 对','5.7 ','章节草稿/5.6-各科学模态对新参与者Q1的预测能力.md'),('5.7 ','5.8 ','章节草稿/5.7-多模态增量、互补与条件预测价值.md'),('附录 D','附录 E','附录/附录D-动作完整结果附表.md'),('附录 E','附录 F','附录/附录E-毫米波心肺完整结果附表.md'),('附录 C','附录 D','附录/附录C-行为完整结果附表.md')]:
        dest=r/'国赛报告'/filename
        assert dest.exists(),dest
        dest.write_text(extract(start,end),encoding='utf-8')
    def append_once(file,marker,body):
        f=r/file;text=f.read_text(encoding='utf-8')
        if marker not in text:f.write_text(text.rstrip()+'\n\n'+marker+'\n\n'+body+'\n',encoding='utf-8')
    append_once('国赛报告/章节草稿/4.1-参与者、研究阶段与样本结构.md','## 正式批次与历史样本边界','正式 116 场的来源登记均为北京，均为两个 B 区块和每场 20 次探针，共 2320 次。珠海不同探针版本以及两阶段预实验没有并入这一分母。第二阶段预实验的 19 份有效记录不等于 19 名独立参与者；原图按记录形成的区间只用于形成性描述。')
    append_once('国赛报告/章节草稿/4.2-SART、思维探针与正式实验程序.md','## 预实验图件的分析单位','第一阶段每区块 216 次，A、B、C 条件分别有 48、24、12 次 No-Go，比例为 22.2%、11.1%、5.6%。报告图 9 的轨迹来自第二阶段 19 份有效记录，历史区间未按重复参与者聚类；图 13 的 20941 次正确 Go 反应与历史图件字节一致，不是正式 116 场的总试次数。两图保留描述用途，并明确其来源。')
    append_once('国赛报告/章节草稿/4.4-科学变量形成、窗口化与质量控制.md','## 当前实际输入的核对补充','实际窗口表的反应时变异系数可估计数为 10 s：2303/2320，20 s：2318/2320，30 s：2318/2320；与每窗口至少 2 个不低于 100 ms 的有效正确 Go 反应条件逐行一致。问卷场次级历史模型另沿用 150 ms 条件，不与探针窗口规则互换。动作输入的 115 场、4903542 条有限记录逐值等于归一化画面变化的每秒速率字段，因此正文 s⁻¹ 正确；历史总账的“无量纲”与导出单位元数据已更正，未改变数值或重训模型。')
    append_once('国赛报告/章节草稿/5.2-行为表现与即时主观状态_20260913.md','## 任务阶段图的单位说明','![反应时变异系数任务进程](../assets/20260928_review_fig17.png)\n\n图中 A 为场次配对均值，B 的横轴为每区块六个连续任务阶段，每段含四个刺激循环；误差线为参与者内汇总后的参与者间标准误，不是 95% 置信区间。模型系数与冻结结果不变。')
    shutil.copy2(a.results/'audit15_fig17.png',r/'国赛报告/assets/20260928_review_fig17.png')
    for filename in ['国赛报告/完整结果/4-Movement动作结果与资产.md']:
        f=r/filename;text=f.read_text(encoding='utf-8').replace('它是无量纲相对指标','它是按帧间隔归一化的画面变化率，单位为 s⁻¹').replace('dimensionless_motion_energy','normalized_motion_energy_per_sec');f.write_text(text,encoding='utf-8')
    append_once('国赛报告/完整结果/4-Movement动作结果与资产.md','## 2026-09-28 单位元数据更正','对照当前生产代码与 115 场实际动作中间表，4903542 条有限值全部来自 `global_motion_energy_per_sec`。先前“无量纲”的总账说明和交接单位标签误用了未除以帧间隔的质量控制字段定义。本次仅更正单位元数据与文稿解释，所有模型数值和冻结输出保持不变；正文每阶段系数仍以 s⁻¹ 表示。详见 09-28-5 逐项修订记录。')
    f=r/'国赛报告/章节草稿/5.5-心肺与注意相关状态.md';text=f.read_text(encoding='utf-8').replace('见 5.6.4 与 5.7','见 5.6.1 与 5.7').replace('该预测使用 110 场、2198 个探针','心肺类别及行为共同比较使用 110 场、2198 个探针，两个独立单指标另使用 2200 个探针').replace('5 个场次、100 个探针窗口','同一参与者的 5 个重复开发场次、100 个探针窗口');f.write_text(text,encoding='utf-8')
    newp=next(p.text for p in d.paragraphs if p.text.startswith('同步心电图（'))
    append_once('国赛报告/章节草稿/5.5-心肺与注意相关状态.md','## 开发误差版本与样本对应',newp+'\n\n报告图 27 为早期 3 场、60 窗口的历史子集，不代表后续冻结 100 窗口的偏差。报告图 15 仅展示 20 s 信号片段与 55 次/分谱峰，不代表正式 30 s 预测窗口或融合估计。')
    append_once('国赛报告/章节草稿/5.9-事后问卷场次级效标一致性.md','## 历史场次模型与图注范围','本节有效正确 Go 反应采用不低于 150 ms 的历史条件，正式探针行为表示另采用不低于 100 ms。图中橙色表示确认性效标一致性、灰色表示探索性模型；颜色不编码单项区间是否排除 1。No-Go 误按率的未校正区间排除 1，但探索性校正后 q = .143。')
    append_once('国赛报告/章节草稿/3-研究目标与总体方案.md','## 界面原型与测量输出边界','报告界面中的状态指数、专注时长、稳定片段、注意画像与心率变异性字段属于原型演示或预留功能。当前二分类模型估计随后报告任务聚焦的概率，尚未建立它与注意强度、专注时间比例及原型可信度的有效换算关系；不将原型呈现当作已验证的测量结果。')
    refs=r/'国赛报告/参考文献.md';text=refs.read_text(encoding='utf-8')
    for ref in m['new_references']:
        if ref not in text:text+='\n'+ref+'\n'
    refs.write_text(text,encoding='utf-8')
    for name in ['source_verification.json','fig17_aggregate.csv','movement_unit.json']:
        shutil.copy2(a.results/('audit15_'+name),r/'运行记录与证据'/('09-28-5-'+name))
    safe={k:v for k,v in m.items() if k not in ['changes','input','output']};safe['changed_regions']=len(m['changes']);safe['local_docx_name']=a.docx.name
    (r/'运行记录与证据/09-28-5-报告修订清单.json').write_text(json.dumps(safe,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Synchronized current chapters, appendices, references and aggregate-only evidence')
if __name__=='__main__':main()
