import cv2, numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
base=mp_python.BaseOptions(model_asset_path='models/face_landmarks/face_landmarker.task')
opts=vision.FaceLandmarkerOptions(base_options=base,running_mode=vision.RunningMode.IMAGE,num_faces=1,output_face_blendshapes=True,output_facial_transformation_matrixes=True,min_face_detection_confidence=0.2)
lm=vision.FaceLandmarker.create_from_options(opts)
import math
for im in ['driver_frontal.jpg','face_open.jpg','face_closed.jpg','driver_face_crop.jpg']:
    img=cv2.imread('test_assets/curated/'+im); h,w=img.shape[:2]
    res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks: continue
    p=np.array([[q.x*w,q.y*h] for q in res.face_landmarks[0]])
    eye=(p[33]+p[263])/2; nose=p[1]; mouth=(p[13]+p[14])/2; facew=np.linalg.norm(p[33]-p[263])
    yaw_proxy=(nose[0]-eye[0])/facew; pitch_proxy=(nose[1]-eye[1])/np.linalg.norm(p[152]-p[10])
    M=np.array(res.facial_transformation_matrixes[0]); R=M[:3,:3]; sy=math.sqrt(R[0,0]**2+R[1,0]**2)
    pitch=math.degrees(math.atan2(R[2,1],R[2,2])); yaw=math.degrees(math.atan2(-R[2,0],sy)); roll=math.degrees(math.atan2(R[1,0],R[0,0]))
    print(im,'yaw_proxy',round(float(yaw_proxy),3),'pitch_proxy',round(float(pitch_proxy),3),'transform yaw/pitch/roll',round(yaw,1),round(pitch,1),round(roll,1))
lm.close()
