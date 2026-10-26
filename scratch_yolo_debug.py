import sys, itertools
sys.path.insert(0, r'C:\D drive\DMS-Model-Testing\src')
import cv2, numpy as np, onnxruntime as ort
from dms.core.geometry import letterbox

model = r'models\candidates\smoking\leviathan_cgr.onnx'
s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
inp = s.get_inputs()[0]; out = s.get_outputs()[0]
print(inp.name, inp.shape, out.name, out.shape)

def stats(name, x):
    raw = s.run([out.name], {inp.name: x})[0]
    arr = raw[0].T if raw.ndim == 3 else raw
    if arr.shape[0] == 5: arr = arr.T
    cls = arr[:, 4:]
    print(name, 'raw', raw.shape, 'min/max', float(raw.min()), float(raw.max()),
          'cls max', float(cls.max()), 'top', [(round(float(v),4), int(np.argmax(arr[i,4:]))) for i,v in zip(np.argsort(cls.max(1))[-5:][::-1], cls.max(1)[np.argsort(cls.max(1))[-5:][::-1]])])

im = cv2.imread(r'test_assets\curated\smoking_positive_1.jpg')
print('img', im.shape)
for size in [640]:
    pad, meta = letterbox(im, (size,size), color=(114,114,114), scale_up=True, stride=32)
    print('letterbox shape', pad.shape, meta)
    for color, cvt in [('rgb', cv2.COLOR_BGR2RGB), ('bgr', None)]:
        img = pad if cvt is None else cv2.cvtColor(pad, cvt)
        x = np.transpose(img.astype(np.float32)/255.0, (2,0,1))[None]
        stats('letterbox '+color, np.ascontiguousarray(x))
    # direct resize
    img = cv2.resize(im, (size,size))
    for color, cvt in [('rgb', cv2.COLOR_BGR2RGB), ('bgr', None)]:
        im2 = img if cvt is None else cv2.cvtColor(img, cvt)
        x = np.transpose(im2.astype(np.float32)/255.0, (2,0,1))[None]
        stats('resize '+color, np.ascontiguousarray(x))
# try ultralytics
try:
    from ultralytics import YOLO
    m = YOLO(model)
    r = m.predict(im, imgsz=640, conf=0.01, verbose=False)[0]
    print('ultralytics boxes', r.boxes.xyxy[:10].numpy(), r.boxes.conf[:10].numpy(), r.boxes.cls[:10].numpy())
except Exception as e:
    print('ultralytics err', type(e).__name__, e)
