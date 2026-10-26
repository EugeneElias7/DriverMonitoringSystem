import requests, hashlib
from pathlib import Path
urls={
 'face_landmarker_official':'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
 'efficientdet_lite0':'https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1.tflite',
}
out=Path('models/_downloads'); out.mkdir(parents=True,exist_ok=True)
for name,u in urls.items():
    try:
        r=requests.get(u,timeout=180,headers={'User-Agent':'dms-research'})
        print(name,r.status_code,len(r.content),r.headers.get('content-type'))
        if r.ok:
            p=out/u.split('/')[-1]; p.write_bytes(r.content)
            print(' sha256',hashlib.sha256(r.content).hexdigest(),'->',p)
    except Exception as e: print(name,'ERR',e)
print('local face hash',hashlib.sha256(Path('models/face_landmarks/face_landmarker.task').read_bytes()).hexdigest())
