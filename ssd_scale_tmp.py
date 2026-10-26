import cv2, numpy as np, tensorflow as tf, json
i=tf.lite.Interpreter(model_path='models/smoking/detect.tflite'); i.allocate_tensors()
ind=i.get_input_details()[0]; outs=i.get_output_details()
score_d=[d for d in outs if d['shape'].tolist()==[1,10]][0]
class_d=[d for d in outs if d['shape'].tolist()==[1,10]][1]
box_d=[d for d in outs if d['shape'].tolist()==[1,10,4]][0]
num_d=[d for d in outs if d['shape'].tolist()==[1]][0]
for name,scale,order in [('0-255 RGB',255,'rgb'),('0-1 RGB',1,'rgb'),('0-255 BGR',255,'bgr'),('0-1 BGR',1,'bgr')]:
    print('\n###',name)
    for im in ['smoking_positive_1.jpg','smoking_negative_1.jpg']:
        img=cv2.imread('test_assets/curated/'+im); h,w=img.shape[:2]
        x=cv2.resize(img,(320,320))
        if order=='rgb': x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        x=(x/scale).astype(np.float32)[None]
        i.set_tensor(ind['index'],x); i.invoke()
        s=i.get_tensor(score_d['index'])[0]; c=i.get_tensor(class_d['index'])[0]; b=i.get_tensor(box_d['index'])[0]; n=int(i.get_tensor(num_d['index'])[0])
        dets=[]
        for k in range(min(n,10)):
            if s[k]>=0.3:
                y1,x1,y2,x2=b[k]; dets.append((int(c[k]),round(float(s[k]),3),[round(float(x1*w)),round(float(y1*h)),round(float(x2*w)),round(float(y2*h))]))
        print(im,dets[:5])
