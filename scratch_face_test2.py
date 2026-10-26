import sys
sys.path.insert(0, r'C:\D drive\DMS-Model-Testing\src')
import cv2
from dms.config import load_config
from dms.detectors.face import FaceAnalyzer
cfg=load_config()
fa=FaceAnalyzer(cfg, mode='IMAGE'); fa.load()
for rel in [r'test_assets\curated\face_open.jpg', r'test_assets\curated\face_closed.jpg', r'test_assets\curated\driver_face_crop.jpg', r'test_assets\curated\driver_frontal.jpg']:
    im=cv2.imread(rel); obs=fa.analyze(im, 1000.0)
    if not obs: print(rel,'NO FACE'); continue
    bs={k:round(v,3) for k,v in obs.blendshapes.items() if 'Blink' in k or 'Squint' in k or 'Wide' in k or 'jaw' in k.lower()}
    print(rel, 'ear',round(obs.ear,3),'pitch',round(obs.pitch,1),'yaw',round(obs.yaw,1), bs)
# test image full and correct roi
im=cv2.imread('test_image.jpg')
for label, roi in [('full',None),('correct_roi',(500,100,900,620))]:
    obs=fa.analyze(im, 1000.0, roi_hint=roi)
    print('test_image',label,'face', None if obs is None else (round(obs.ear,3), round(obs.mar,3), round(obs.pitch,1), round(obs.yaw,1), obs.pose_source, obs.roi_upscaled))
fa.close()
