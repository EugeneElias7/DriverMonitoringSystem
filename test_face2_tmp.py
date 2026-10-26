import cv2, numpy as np, requests, time
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
img=cv2.imread('test_image.jpg')
Path('test_assets').mkdir(exist_ok=True)
# Official MediaPipe sample image
for name,url in [('portrait.jpg','https://storage.googleapis.com/mediapipe-assets/portrait.jpg')]:
    r=requests.get(url,timeout=60)
    print(name, r.status_code, len(r.content))
    if r.ok: Path('test_assets',name).write_bytes(r.content)
base=mp_python.BaseOptions(model_asset_path='models/face_landmarks/face_landmarker.task')
for tag,im in [('full',img),('crop',img[150:750,450:900])]:
    if im.size==0: continue
    opts=vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2, min_tracking_confidence=0.2)
    lm=vision.FaceLandmarker.create_from_options(opts)
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    print(tag, im.shape, 'faces',len(res.face_landmarks))
    if res.face_landmarks:
        print(' blink', {c.category_name:round(c.score,3) for c in res.face_blendshapes[0] if 'Blink' in c.category_name})
        out=im.copy(); h,w=im.shape[:2]
        for p in res.face_landmarks[0]: cv2.circle(out,(int(p.x*w),int(p.y*h)),1,(0,255,0),-1)
        cv2.imwrite(f'results/face_{tag}.jpg',out)
    lm.close()
if Path('test_assets/portrait.jpg').exists():
    im=cv2.imread('test_assets/portrait.jpg')
    opts=vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    lm=vision.FaceLandmarker.create_from_options(opts)
    res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
    print('portrait',im.shape,'faces',len(res.face_landmarks))
    if res.face_landmarks: print(' blink', {c.category_name:round(c.score,3) for c in res.face_blendshapes[0] if 'Blink' in c.category_name})
    lm.close()
