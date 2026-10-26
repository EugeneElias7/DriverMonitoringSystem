import cv2, numpy as np, time
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
print('mediapipe', mp.__version__)
model=str(Path('models/face_landmarks/face_landmarker.task'))
base=mp_python.BaseOptions(model_asset_path=model)
opts=vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3)
lm=vision.FaceLandmarker.create_from_options(opts)
img=cv2.imread('test_image.jpg'); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mpimg=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
t=time.perf_counter(); res=lm.detect(mpimg); dt=time.perf_counter()-t
print('detect ms',dt*1000,'faces',len(res.face_landmarks))
if res.face_landmarks:
    pts=np.array([[p.x,p.y,p.z] for p in res.face_landmarks[0]])
    print('landmarks',pts.shape,'x range',pts[:,0].min(),pts[:,0].max(),'y range',pts[:,1].min(),pts[:,1].max())
    names=[c.category_name for c in res.face_blendshapes[0]]
    scores=[c.score for c in res.face_blendshapes[0]]
    for n,s in zip(names,scores):
        if 'Blink' in n or 'Brow' in n or 'Jaw' in n or 'Mouth' in n: print(f'{n}: {s:.3f}')
    print('matrix shape',np.array(res.facial_transformation_matrixes[0]).shape)
    print(np.array(res.facial_transformation_matrixes[0]))
    h,w=img.shape[:2]
    for p in pts:
        cv2.circle(img,(int(p[0]*w),int(p[1]*h)),1,(0,255,0),-1)
    cv2.imwrite('results/face_landmarker_test.jpg',img)
    print('saved results/face_landmarker_test.jpg')
lm.close()
