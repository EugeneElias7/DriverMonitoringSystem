import sys, time
sys.path.insert(0, r'C:\D drive\DMS-Model-Testing\src')
import cv2
from dms.config import load_config
from dms.backends.yolo_onnx import YoloOnnxBackend

cases = [
    ('leviathan_cgr.onnx', ['cigarette']),
    ('meetcool_yolov8n.onnx', ['cigarette', 'person', 'smoke']),
]
imgs = [
    r'test_assets\curated\smoking_positive_1.jpg',
    r'test_assets\curated\smoking_positive_2.jpg',
    r'test_assets\curated\smoking_cigarette_in_hand.jpg',
    r'test_assets\curated\smoking_negative_1.jpg',
    r'test_assets\curated\smoking_negative_2.jpg',
    r'test_image.jpg',
]
for model, names in cases:
    cfg = load_config(overrides={'smoking.model_path': 'models/candidates/smoking/' + model, 'smoking.class_names': names})
    b = YoloOnnxBackend('test', cfg, cfg.resolve_path(cfg.get('smoking.model_path')))
    try:
        t0=time.time(); b.load(); print('\n###', model, round((time.time()-t0)*1000,1), 'ms', b.health.to_dict(), b.class_names)
    except Exception as e:
        print('\n###', model, 'LOAD FAILED', type(e).__name__, e); continue
    for rel in imgs:
        im=cv2.imread(rel)
        t0=time.time(); ds=b.detect(im, 1000); ms=(time.time()-t0)*1000
        print(rel.split('\\')[-1], f'{ms:.1f}ms', [(d.label, round(d.confidence,3), [round(v) for v in d.bbox]) for d in ds[:5]])
    b.close()
