import requests
from pathlib import Path
urls=['https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v2_1_metadata_1.tflite','https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_ssd_mobilenet_v1_1_metadata_1.tflite','https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/1/default/1?lite-format=tflite','https://www.kaggle.com/api/v1/models/tensorflow/ssd-mobilenet-v2/tfLite/fpnlite-320-metadata/1/download','https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip']
for u in urls:
    try:
        r=requests.get(u,timeout=120,headers={'User-Agent':'dms-research'},allow_redirects=True)
        print(u[:120],r.status_code,len(r.content),r.headers.get('content-type'),r.url[:100])
        if r.ok and len(r.content)>100000:
            p=Path('models/_downloads')/(u.split('/')[-1].split('?')[0]); p.write_bytes(r.content); print(' ->',p)
    except Exception as e: print('ERR',e)
