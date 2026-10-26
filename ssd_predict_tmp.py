import cv2, numpy as np, tensorflow as tf, json
from pathlib import Path
p='models/smoking/detect.tflite'
i=tf.lite.Interpreter(model_path=p); i.allocate_tensors()
ind=i.get_input_details()[0]; outs=i.get_output_details()
# identify by shape
byd={tuple(d['shape']):d for d in outs}
print('outs',[(d['name'],d['shape']) for d in outs])
for im in ['smoking_positive_1.jpg','smoking_positive_2.jpg','smoking_negative_1.jpg','smoking_negative_2.jpg','smoking_negative_3.jpg','smoking_hard_1.jpg']:
    img=cv2.imread('test_assets/curated/'+im); h,w=img.shape[:2]
    x=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(320,320)).astype(np.float32)[None]
    i.set_tensor(ind['index'],x); i.invoke()
    scores=i.get_tensor(byd[(1,10)]['index'])[0]
    classes=i.get_tensor([d for d in outs if d['name']=='StatefulPartitionedCall:2'][0]['index'])[0]
    boxes=i.get_tensor(byd[(1,10,4)]['index'])[0]
    n=int(i.get_tensor(byd[(1,)]['index'])[0])
    dets=[]
    for k in range(min(n,10)):
        if scores[k]>=0.15:
            y1,x1,y2,x2=boxes[k]
            dets.append({'c':int(classes[k]),'p':round(float(scores[k]),3),'xyxy':[round(float(x1*w),1),round(float(y1*h),1),round(float(x2*w),1),round(float(y2*h),1)]})
    print(im,json.dumps(dets))
