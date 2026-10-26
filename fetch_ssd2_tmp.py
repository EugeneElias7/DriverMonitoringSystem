import requests
from pathlib import Path
cands=[
'https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/object_detection/android/app/src/main/assets/ssd_mobilenet_v2.tflite',
'https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/object_detection/ios/ObjectDetection/Model/ssd_mobilenet_v2.tflite',
'https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/object_detection/raspberry_pi/ssd_mobilenet_v2.tflite',
'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v2_1_metadata_2.tflite?x=1',
'https://github.com/tensorflow/examples/raw/master/lite/examples/object_detection/android/app/src/main/assets/efficientdet_lite0.tflite',
]
for u in cands:
    try:
        r=requests.get(u,timeout=60,headers={'User-Agent':'dms-research'},allow_redirects=True)
        print(r.status_code,len(r.content),r.headers.get('content-type'),u[:110])
        if r.ok and len(r.content)>100000: print(' FOUND',u)
    except Exception as e: print('ERR',e)
