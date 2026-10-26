import sys, time
sys.path.insert(0, r'C:\D drive\DMS-Model-Testing\src')
import cv2, numpy as np
from dms.config import load_config
from dms.backends.tflite import TFLiteDetectorBackend

cfg=load_config()
for name, model, labels, rng, known in [
    ('existing', 'models/smoking/detect.tflite', 'models/smoking/labelmap.txt', 'unit', True),
    ('existing_raw', 'models/smoking/detect.tflite', 'models/smoking/labelmap.txt', 'raw', True),
    ('eff', 'models/_downloads/lite-model_efficientdet_lite0_detection_metadata_1.tflite', 'models/_downloads/efficientdet_labels.txt', 'raw', False),
    ('ssd', 'models/_downloads/ssd_v1/detect.tflite', 'models/_downloads/ssd_v1/labelmap.txt', 'raw', False),
]:
    b=TFLiteDetectorBackend(name, cfg, cfg.resolve_path(model), cfg.resolve_path(labels), input_range=rng, score_threshold=0.3, known_issue=known)
    try:
        t0=time.time(); b.load(); print('\n###',name,'load',round((time.time()-t0)*1000),'ms',b.health.to_dict(),'classes',len(b.class_names))
    except Exception as e:
        print('\n###',name,'LOAD FAIL',type(e).__name__,e); continue
    for rel in [r'test_assets\curated\smoking_positive_1.jpg', r'test_assets\curated\smoking_positive_2.jpg', r'test_assets\curated\smoking_negative_1.jpg', r'test_image.jpg']:
        im=cv2.imread(rel)
        t0=time.time(); ds=b.detect(im,1000); ms=(time.time()-t0)*1000
        print(rel.split('\\')[-1], f'{ms:.1f}ms', len(ds), [(d.label, round(d.confidence,3), [round(v) for v in d.bbox], d.class_id) for d in ds[:6]])
    b.close()
