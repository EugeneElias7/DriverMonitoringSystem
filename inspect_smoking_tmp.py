from pathlib import Path
import cv2, numpy as np, tensorflow as tf
p=Path('models/smoking/detect.tflite')
img=cv2.cvtColor(cv2.imread('test_image.jpg'), cv2.COLOR_BGR2RGB)
h,w=img.shape[:2]
i=tf.lite.Interpreter(model_path=str(p)); i.allocate_tensors()
ind=i.get_input_details()[0]; outs=i.get_output_details()
x=cv2.resize(img,(ind['shape'][2],ind['shape'][1])).astype(np.float32)[None,...]
i.set_tensor(ind['index'],x); i.invoke()
print('input pixel range', x.min(), x.max())
for d in outs:
    v=i.get_tensor(d['index'])
    print(d['name'], d['shape'], 'min',float(v.min()),'max',float(v.max()),'sample',v.reshape(-1)[:12])
run=i.get_signature_runner('serving_default')
res=run(input=x)
for k,v in res.items(): print('SIG',k, v.shape, v.reshape(-1)[:8])
