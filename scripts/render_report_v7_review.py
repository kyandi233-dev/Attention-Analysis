"""Rasterize native Word PDF and prepare visual review sheets."""
import argparse,json,sys
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
sys.stdout.reconfigure(encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--pdf',default='v7_render.pdf');a=ap.parse_args();root=a.folder;tag=Path(a.pdf).stem;dest=root/('render' if tag=='v7_render' else tag);dest.mkdir(exist_ok=True)
def contacts(items,prefix,cols,rows,size):
 for off in range(0,len(items),cols*rows):
  canvas=Image.new('RGB',(cols*size[0],rows*size[1]),'#bbb')
  for j,(label,pic) in enumerate(items[off:off+cols*rows]):
   pic=pic.copy();pic.thumbnail((size[0]-16,size[1]-30));tile=Image.new('RGB',size,'white');tile.paste(pic,((size[0]-pic.width)//2,25));ImageDraw.Draw(tile).text((8,7),label,fill='black');canvas.paste(tile,((j%cols)*size[0],(j//cols)*size[1]))
  canvas.save(dest/f'{prefix}-{off//(cols*rows)+1:02}.jpg')
pdf=pdfium.PdfDocument(str(root/a.pdf));texts=[];thumbs=[]
for i,p in enumerate(pdf):
 pic=p.render(scale=1.5).to_pil().convert('RGB');pic.save(dest/f'page-{i+1:03}.png');thumbs.append((str(i+1),pic));texts.append(p.get_textpage().get_text_range())
contacts(thumbs,'pages',4,3,(280,405));contacts([(p.stem,Image.open(p).convert('RGB')) for p in sorted(root.glob('fig*_v7.png'))],'plots',2,3,(650,390))
(root/('render_text.txt' if tag=='v7_render' else tag+'_text.txt')).write_text('\n\n'.join(f'PAGE {i+1}\n{s}' for i,s in enumerate(texts)),encoding='utf-8')
print('PAGES',len(pdf));print('EMPTY_PAGES',[i+1 for i,t in enumerate(texts) if not t.strip()]);print('CAPTION_PAGES',[(i+1,t[:60].replace('\n',' ')) for i,t in enumerate(texts) if '表 A1' in t or '表 F1' in t])
