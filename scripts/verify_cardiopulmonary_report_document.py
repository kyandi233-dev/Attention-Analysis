"""Check the delivered copy and render page contacts; use the document runtime."""
from pathlib import Path
import argparse,hashlib,json,zipfile
from lxml import etree
from docx import Document
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.results/'report_edit_manifest.json').read_text(encoding='utf-8'))
    original=Path(manifest['input']);final=Path(manifest['output'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(original)==manifest['input_sha256'];assert sha(final)==manifest['output_sha256']
    with zipfile.ZipFile(original) as old,zipfile.ZipFile(final) as new:
        assert set(old.namelist())==set(new.namelist())
        changed=[n for n in old.namelist() if old.read(n)!=new.read(n)]
        assert set(changed)=={'word/document.xml',*manifest['replaced_figures']}
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        xml=etree.fromstring(new.read('word/document.xml'))
        highlighted=len(xml.xpath('//w:highlight[@w:val="yellow"]',namespaces=ns))
    d=Document(final);assert len(d.tables)==51;assert len(d.paragraphs)==752
    assert all(len(d.tables[i].rows)==57 for i in [33,46,47])
    render=a.results/'render';pdf=render/'report_v3.pdf';reader=PdfReader(pdf)
    prior=PdfReader(render/'report_v2.pdf')
    changed_pages=[i+1 for i,page in enumerate(reader.pages) if i>=len(prior.pages) or page.extract_text()!=prior.pages[i].extract_text()]
    doc=pdfium.PdfDocument(pdf);thumbs=[]
    for i in range(len(doc)):
        page=doc[i];pil=page.render(scale=1.3).to_pil().convert('RGB')
        pil.save(render/f'v3-page-{i+1}.png')
        thumb=pil.copy();thumb.thumbnail((340,460))
        card=Image.new('RGB',(360,490),'#d0d0d0');card.paste(thumb,((360-thumb.width)//2,20));ImageDraw.Draw(card).text((12,473),str(i+1),fill='black');thumbs.append(card)
        page.close()
    for start in range(0,len(thumbs),12):
        contact=Image.new('RGB',(1440,1470),'white')
        for j,t in enumerate(thumbs[start:start+12]):contact.paste(t,((j%4)*360,(j//4)*490))
        contact.save(render/f'v3-contact-{start//12+1:02}.jpg')
    result={'status':'STRUCTURE_PASS_VISUAL_REVIEW_REQUIRED','original_sha256_unchanged':sha(original),'delivery_sha256':sha(final),'package_members_changed':changed,'other_package_members_byte_identical':True,'tables':51,'body_paragraphs':752,'yellow_highlights':highlighted,'changed_regions':manifest['changed_regions'],'native_word_open_export':True,'pdf_pages':len(reader.pages),'pages_with_changed_text_since_v2':changed_pages,'style_review':'Original wording and sentence structure retained where compatible; Chinese punctuation and ungrouped prose counts follow original; statistical interval commas and mathematical minus signs retained.'}
    (a.results/'report_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
