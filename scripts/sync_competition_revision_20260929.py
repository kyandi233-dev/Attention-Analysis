"""Synchronize competition prose separately from aggregate audit evidence."""
from pathlib import Path
import argparse,json,re,shutil
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--formal-repo',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);a=ap.parse_args();r=a.formal_repo
    m=json.loads((a.results/'v5_edit_manifest.json').read_text(encoding='utf-8'));d=Document(m['output'])
    chapters=r/'国赛报告/章节草稿';assets=r/'国赛报告/assets';modified=[]
    remove_headings=['正式批次与历史样本边界','预实验图件的分析单位','当前实际输入的核对补充','任务阶段图的单位说明','开发误差版本与样本对应','历史场次模型与图注范围','界面原型与测量输出边界']
    for f in list(chapters.glob('*.md'))+list((r/'国赛报告/附录').glob('*.md')):
        text=old=f.read_text(encoding='utf-8')
        for heading in remove_headings:text=re.sub(r'\n## '+re.escape(heading)+r'\n.*?(?=\n## |\Z)','',text,flags=re.S)
        for change in m['changes']:
            before,after=change['before'],change['after']
            if len(before)>24 and change['location'].startswith(('paragraph ','delete paragraph ')):text=text.replace(before,after)
        text=text.replace('无已检测时序歧义遗漏','筛选后遗漏').replace('时序歧义遗漏','伴随提前或延续按键的遗漏')
        if text!=old:f.write_text(text,encoding='utf-8');modified.append(str(f.relative_to(r)))
    def extract(start,end):
        active=False;out=[]
        for el in d._element.body:
            if el.tag==qn('w:p'):
                p=Paragraph(el,d);text=p.text.strip()
                if active and text.startswith(end):break
                if text.startswith(start):active=True
                if not active:continue
                blips=p._p.xpath('.//a:blip')
                if blips:
                    part=d.part.related_parts[blips[0].get(qn('r:embed'))];name=Path(str(part.partname)).name
                    # Reuse identical assets instead of duplicating older figures.
                    dest=next((f for f in assets.glob('*'+Path(name).suffix) if f.read_bytes()==part.blob),None)
                    if dest is None:dest=assets/('20260929_competition_'+name);dest.write_bytes(part.blob)
                    out.append(f'![研究结果图](../assets/{dest.name})')
                elif text:out.append(('# ' if text.startswith(start) else '## ' if re.match(r'^(\d\.\d(?:\.\d)?|[A-G]\.\d)\s',text) else '')+text)
            elif el.tag==qn('w:tbl') and active:
                t=Table(el,d);rows=['| '+' | '.join(c.text.replace('\n','<br>').replace('|','\\|') for c in row.cells)+' |' for row in t.rows]
                rows.insert(1,'| '+' | '.join('---' for _ in t.rows[0].cells)+' |');out.append('\n'.join(rows))
        assert len(out)>2,start
        return '\n\n'.join(out)+'\n'
    pairs=[('5.1 ','5.2 ','5.1-分析样本、数据覆盖与测量质量_20260913.md'),('5.2 ','5.3 ','5.2-行为表现与即时主观状态_20260913.md'),('5.3 ','5.4 ','5.3-眼部测量资格与瞳孔动态_20260913.md'),('5.4 ','5.5 ','5.4-动作与注意相关状态_20260913.md'),('5.5 ','5.6 ','5.5-心肺与注意相关状态.md'),('5.6 ','5.7 ','5.6-各科学模态对新参与者Q1的预测能力.md'),('5.7 ','5.8 ','5.7-多模态增量、互补与条件预测价值.md'),('5.8 ','5.9 ','5.8-设备组合与系统级输出.md'),('5.9 ','5.10 ','5.9-事后问卷场次级效标一致性.md'),('5.10 ','6 ','5.10-测评系统的界面与交互设计.md'),('6 ','参考文献','6-分析与讨论.md'),('4.6 ','5 ','4.6-跨参与者监督学习与多模态比较.md')]
    for start,end,name in pairs:
        f=chapters/name;f.write_text(extract(start,end),encoding='utf-8');modified.append(str(f.relative_to(r)))
    (chapters/'摘要与关键词.md').write_text(extract('摘要','1 引言'),encoding='utf-8')
    f=chapters/'4.4-科学变量形成、窗口化与质量控制.md'
    f.write_text(re.sub(r'(?m)^(#+ )4\.5',r'\g<1>4.4',extract('4.5 ','4.6 ')),encoding='utf-8');modified.append(str(f.relative_to(r)))
    for letter,nxt,name in [('C','D','行为'),('D','E','动作'),('E','F','毫米波心肺')]:
        f=r/f'国赛报告/附录/附录{letter}-{name}完整结果附表.md';assert f.exists()
        f.write_text(extract('附录 '+letter,'附录 '+nxt),encoding='utf-8');modified.append(str(f.relative_to(r)))
    f=chapters/'4.2-SART、思维探针与正式实验程序.md';text=f.read_text(encoding='utf-8')
    p=next(p.text for p in d.paragraphs if p.text.startswith('第二阶段的 19 份'))
    text=re.sub(r'第二阶段采用 3 个结构相同的区块.*?(?=\n\n)',p,text,flags=re.S);f.write_text(text,encoding='utf-8')
    for name,out in [('v5_descriptive_results.json','重算描述统计.json'),('v5_document_verification.json','文档验收.json'),('v5_fig17_aggregate.csv','任务阶段聚合.csv')]:shutil.copy2(a.results/name,r/'运行记录与证据'/('09-29-1-'+out))
    safe={k:v for k,v in m.items() if k not in ['changes','input','output']};safe['local_docx_name']=Path(m['output']).name;safe['modified_entries']=sorted(set(modified))
    (r/'运行记录与证据/09-29-1-修订清单.json').write_text(json.dumps(safe,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Updated competition chapters and aggregate-only evidence; no model fitting.')

if __name__=='__main__':main()
