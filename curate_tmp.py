from pathlib import Path
from PIL import Image
import shutil, json
src=Path('test_assets/raw'); out=Path('test_assets/curated'); out.mkdir(parents=True,exist_ok=True)
def cp(srcname,dst):
    p=src/srcname
    if p.exists():
        im=Image.open(p).convert('RGB'); im.save(out/dst,quality=92); print(dst,im.size)
    else: print('MISSING',srcname)
cp('Woman_knitwear_car_Unsplash_.jpg','driver_frontal.jpg')
cp('portrait.jpg','face_open.jpg') if (src/'portrait.jpg').exists() else cp('../portrait.jpg','face_open.jpg')
cp('A_bearded_man_head_and_shoulders_with_his_eyes_closed._Pho_Wellcome_V0029461.jpg','face_closed.jpg')
cp('Woman_with_hand-rolled_cigarette.jpg','smoking_positive_1.jpg')
cp('Portrait_man_smoking_cigarette_Fortepan_11621.jpg','smoking_positive_2.jpg')
cp('Fashion%20Time%2C%20Modeling%20Photography%20in%20Tbilisi%20-%20Irania.jpg','smoking_cigarette_in_hand.jpg')
cp('Series%20of%20three%20photos%20showing%20a%20hysterical%20yawning%20wo.jpg','face_yawning_triptych.jpg')
cp('Infrared%20portrait%20comparison.jpg.jpg','ir_comparison.jpg')
# crops
im=Image.open(out/'face_yawning_triptych.jpg'); w,h=im.size
im.crop((int(w*0.34),int(h*0.02),int(w*0.66),int(h*0.95))).save(out/'face_yawning.jpg',quality=92)
print('yawning',Image.open(out/'face_yawning.jpg').size)
im=Image.open(out/'ir_comparison.jpg'); w,h=im.size
print('ir comparison',w,h)
# panels roughly thirds horizontally; crop NIR (middle) and visible (left)
panel_w=w/3
for name,x0 in [('ir_visible_panel',0),('ir_nir_panel',1),('ir_swir_panel',2)]:
    im.crop((int(x0*panel_w),0,int((x0+1)*panel_w),h)).save(out/f'{name}.jpg',quality=95)
# face crop from NIR panel; inspect manually later
nir=Image.open(out/'ir_nir_panel.jpg'); nw,nh=nir.size
nir.crop((int(nw*0.32),int(nh*0.05),int(nw*0.72),int(nh*0.48))).resize((640,640)).save(out/'ir_face.jpg',quality=95)
vis=Image.open(out/'ir_visible_panel.jpg'); vw,vh=vis.size
vis.crop((int(vw*0.32),int(vh*0.05),int(vw*0.72),int(vh*0.48))).resize((640,640)).save(out/'ir_visible_face.jpg',quality=95)
# driver wide crop
from PIL import Image as I
d=I.open('test_image.jpg'); d.crop((450,150,900,750)).resize((900,1200)).save(out/'driver_face_crop.jpg',quality=92)
print('done')
for p in sorted(out.glob('*')): print(p.name,p.stat().st_size)
