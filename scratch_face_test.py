import sys, time
from pathlib import Path
sys.path.insert(0, r'C:\D drive\DMS-Model-Testing\src')
import cv2
from dms.config import load_config
from dms.detectors.face import FaceAnalyzer

cfg = load_config()
paths = [
    r'test_assets\curated\driver_frontal.jpg',
    r'test_assets\curated\face_open.jpg',
    r'test_assets\curated\face_closed.jpg',
    r'test_assets\curated\face_yawning.jpg',
    r'test_assets\curated\ir_face.jpg',
    r'test_assets\curated\driver_face_crop.jpg',
    r'test_image.jpg',
]
fa = FaceAnalyzer(cfg, mode='IMAGE')
t0 = time.time(); fa.load(); print('load_ms', (time.time()-t0)*1000, fa.health.to_dict())
for rel in paths:
    im = cv2.imread(rel)
    if im is None:
        print('MISSING', rel); continue
    if rel.endswith('test_image.jpg'):
        roi = (150, 450, 750, 900)
        crop = im[450:900, 150:750]
    else:
        roi = None
        crop = im
    t0 = time.time()
    obs = fa.analyze(crop, 1000.0)
    ms = (time.time()-t0)*1000
    if obs is None:
        print(Path(rel).name, im.shape, 'NO FACE', round(ms, 1)); continue
    print(Path(rel).name, im.shape, 'face', [round(x, 1) for x in obs.bbox], 'ear', round(obs.ear, 4), 'mar', round(obs.mar, 3),
          'pose', tuple(round(x, 1) for x in (obs.pitch, obs.yaw, obs.roll)), obs.pose_source, obs.roi, round(ms, 1), 'ms')
fa.close()
