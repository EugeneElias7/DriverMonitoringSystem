from pathlib import Path
from PIL import Image, ImageDraw
import json, math
files=sorted(Path('test_assets/raw').glob('*'))
print(len(files))
for i,f in enumerate(files):
    try:
        im=Image.open(f); print(i, f.name[:110], im.size, im.mode)
    except Exception as e: print(i,f.name,'ERR',e)
# contact sheets
def sheet(files, out, cols=5, cell=260):
    rows=math.ceil(len(files)/cols); W=cols*cell; H=rows*(cell+30)
    canvas=Image.new('RGB',(W,H),(20,20,20)); d=ImageDraw.Draw(canvas)
    for i,f in enumerate(files):
        try:
            im=Image.open(f).convert('RGB'); im.thumbnail((cell-10,cell-10))
            x=(i%cols)*cell; y=(i//cols)*(cell+30)
            canvas.paste(im,(x+5,y+5)); d.text((x+5,y+cell+2),f"{i}:{f.stem[:30]}",fill=(255,255,0))
        except Exception: pass
    canvas.save(out,quality=85)
for k in range(0,len(files),25):
    sheet(files[k:k+25],f'test_assets/contact_{k//25}.jpg')
print('sheets written')
