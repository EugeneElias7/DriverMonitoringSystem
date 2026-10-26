import requests, hashlib
from pathlib import Path
out=Path('models/_downloads'); out.mkdir(parents=True,exist_ok=True)
urls=[
'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite',
'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v2_1_metadata_2.tflite',
'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_2.tflite',
'https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/ssd_mobilenet_v2/1/default/1.tflite',
'https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite0/detection/1.tflite',
]
for u in urls:
    try:
        r=requests.get(u,timeout=120,headers={'User-Agent':'dms-research'})
        print(u.split('/')[-1],r.status_code,len(r.content),r.headers.get('content-type'))
        if r.ok and len(r.content)>100000:
            p=out/u.split('/')[-1]; p.write_bytes(r.content); print(' sha256',hashlib.sha256(r.content).hexdigest())
    except Exception as e: print('ERR',e)
