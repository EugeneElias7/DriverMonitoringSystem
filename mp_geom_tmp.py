import cv2, numpy as np, json, math
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
base=mp_python.BaseOptions(model_asset_path='models/face_landmarks/face_landmarker.task')
opts=vision.FaceLandmarkerOptions(base_options=base,running_mode=vision.RunningMode.IMAGE,num_faces=1,output_face_blendshapes=True,output_facial_transformation_matrixes=True,min_face_detection_confidence=0.2,min_face_presence_confidence=0.2,min_tracking_confidence=0.2)
lm=vision.FaceLandmarker.create_from_options(opts)
LEFT=[33,160,158,133,153,144]; RIGHT=[362,385,387,263,373,380]
def d(a,b): return float(np.linalg.norm(a-b))
def ear(pts,idx):
    p=[pts[i] for i in idx]; return (d(p[1],p[5])+d(p[2],p[4]))/(2*d(p[0],p[3])+1e-9)
for im in ['driver_frontal.jpg','face_open.jpg','face_closed.jpg','face_yawning.jpg','smoking_positive_1.jpg','smoking_positive_2.jpg','smoking_hard_1.jpg','ir_face.jpg','ir_visible_face.jpg','driver_face_crop.jpg']:
    img=cv2.imread('test_assets/curated/'+im); h,w=img.shape[:2]
    res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
    print('\n###',im,img.shape,'faces',len(res.face_landmarks))
    if not res.face_landmarks: continue
    pts=np.array([[p.x*w,p.y*h,p.z*w] for p in res.face_landmarks[0]])
    le=ear(pts,LEFT); re=ear(pts,RIGHT)
    mar=d(pts[13],pts[14])/(d(pts[61],pts[291])+1e-9)
    bbox=[pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max()]
    bl={c.category_name:round(c.score,3) for c in res.face_blendshapes[0] if 'Blink' in c.category_name or 'Jaw' in c.category_name}
    print('bbox',[round(float(x),1) for x in bbox],'earL',round(le,3),'earR',round(re,3),'mar',round(mar,3),'blink',bl)
    M=np.array(res.facial_transformation_matrixes[0]); print('M',np.round(M,3).tolist())
    R=M[:3,:3]
    # common convention: pitch around x, yaw around y, roll around z
    sy=math.sqrt(R[0,0]**2+R[1,0]**2)
    pitch=math.atan2(R[2,1],R[2,2]); yaw=math.atan2(-R[2,0],sy); roll=math.atan2(R[1,0],R[0,0])
    print('euler_deg',[round(math.degrees(v),1) for v in (yaw,pitch,roll)])
    out=img.copy()
    for p in pts: cv2.circle(out,(int(p[0]),int(p[1])),1,(0,255,0),-1)
    cv2.imwrite('results/mesh_'+im,out)
lm.close()
