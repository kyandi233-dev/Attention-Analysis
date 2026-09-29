"""Restore inherited Word Caption formatting in the v7 report without touching results."""
import argparse,copy,hashlib,json,re,zipfile
from pathlib import Path
from lxml import etree
from docx import Document
from docx.oxml.ns import qn
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);a=ap.parse_args()
assert a.source.exists() and a.reference.exists()
m=json.loads(a.manifest.read_text(encoding='utf-8'));assert hashlib.sha256(a.source.read_bytes()).hexdigest()==m['output_sha256']
src=Document(a.source);ref=Document(a.reference)
out=a.source.with_name('FocusWave_国赛报告_题注格式修复_标黄副本_20260929_v8.docx')
oldfig={int(k):v for k,v in m['figure_number_map'].items()};newold={v:k for k,v in oldfig.items()}
ref_fig={int(re.match(r'图\s*(\d+)',p.text)[1]):p for p in ref.paragraphs if re.match(r'^图\s*\d+',p.text) and p.style.name=='Caption'}
ref_tab={int(re.match(r'表\s*(\d+)',p.text)[1]):p for p in ref.paragraphs if re.match(r'^表\s*\d+',p.text) and p.style.name=='Caption'}
assert len(ref_fig)==45 and len(ref_tab)==21
ledger=[]
for p in src.paragraphs:
 if p.style.name!='Caption':continue
 match=re.match(r'^(图|表)\s*([A-F]?)(\d+)\s*(.*)',p.text)
 assert match,p.text
 kind,letter,number,title=match.groups();number=int(number)
 if kind=='图':
  old=newold[number];original=ref_fig[old];changed_number=number!=old;new_caption=False
 elif letter:
  original=ref_tab[1];changed_number=True;new_caption=True
 else:
  original=ref_tab[number];changed_number=False;new_caption=False
 original_title=re.sub(r'^(图|表)\s*\d+\s*','',original.text)
 title_changed=new_caption or title!=original_title
 # Reuse the reference's actual paragraph properties; the Caption style supplies
 # alignment, 9-point bold typography, line spacing, and keep-with-next.
 old_ppr=p._p.pPr
 if old_ppr is not None:p._p.remove(old_ppr)
 if original._p.pPr is not None:p._p.insert(0,copy.deepcopy(original._p.pPr))
 for r in p.runs:
  rp=r._r.rPr
  if rp is not None:r._r.remove(rp)
  # Mark only genuinely changed caption text. Existing titles inherit the
  # reference's typography and do not become yellow as a whole.
  mark=False
  if new_caption:mark=True
  elif r.text.strip()==str(number) and changed_number:mark=True
  elif r.text.strip()==title and title_changed:mark=True
  if mark:
   rp=etree.Element(qn('w:rPr'));h=etree.SubElement(rp,qn('w:highlight'));h.set(qn('w:val'),'yellow');r._r.insert(0,rp)
 ledger.append({'kind':kind,'original_number':old if kind=='图' else None,'number':f'{letter}{number}','title':title,'new_caption':new_caption,'number_changed':changed_number,'title_changed':title_changed})
assert len(ledger)==86
revisions={
 '四类注意内容的分别预测使用独立的多分类训练与验证，并评价各类别的区分表现与概率可靠性，结果见 5.6.5。二分类与多分类分别对应合并类别和细分内容的预测目标。':'本报告以任务聚焦与其他注意内容的二分类预测为主；各类注意内容之间的差异由解释性关联分析单独呈现。',
 '困倦—清醒报告和事后问卷用于解释性分析。逐探针预测以注意内容为目标，四类内容的分别预测通过独立的多分类分析评价。':'困倦—清醒报告和事后问卷用于解释性分析；逐探针预测以是否报告任务聚焦为目标。'
}
for old,new in revisions.items():
 matches=[p for p in src.paragraphs if p.text==old];assert len(matches)==1,(old,len(matches))
 p=matches[0]
 for child in list(p._p):
  if child.tag!=qn('w:pPr'):p._p.remove(child)
 r=p.add_run(new);rp=etree.Element(qn('w:rPr'));h=etree.SubElement(rp,qn('w:highlight'));h.set(qn('w:val'),'yellow');r._r.insert(0,rp)
with zipfile.ZipFile(a.source) as z,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dst:
 for info in z.infolist():
  blob=etree.tostring(src._element,xml_declaration=True,encoding='UTF-8',standalone=True) if info.filename=='word/document.xml' else z.read(info.filename)
  dst.writestr(info,blob)
re=Document(out)
assert len([p for p in re.paragraphs if p.style.name=='Caption'])==86
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
log={'input':str(a.source),'input_sha256':sha(a.source),'reference':str(a.reference),'reference_sha256':sha(a.reference),'output':str(out),'output_sha256':sha(out),'captions':ledger,'prose_corrections':revisions,'changes_only':['word/document.xml'],'models_refitted':0}
logpath=a.manifest.parent/'caption_v8_manifest.json';logpath.write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(out),'sha256':log['output_sha256'],'captions':len(ledger)},ensure_ascii=False))
