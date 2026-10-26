from ultralytics import YOLO
from pathlib import Path
import json
models=['meetcool_yolov8n.pt','notacodinggeek_yolov8n.pt','cadilak_yolov8_2class.pt','abhishek_yolo_smoking.pt','beehzod_yolo11m_cigarette2.pt','beehzod_yolo11m_finetuned.pt','leviathan_cgr.onnx']
imgs=['smoking_positive_1.jpg','smoking_positive_2.jpg','smoking_negative_1.jpg','smoking_negative_2.jpg','smoking_negative_3.jpg','smoking_hard_1.jpg']
for mn in models:
    m=YOLO(str(Path('models/candidates/smoking')/mn))
    print('\n########',mn,m.names)
    for im in imgs:
        r=m.predict('test_assets/curated/'+im,imgsz=640,conf=0.15,iou=0.5,verbose=False,device='cpu')[0]
        dets=[]
        if r.boxes is not None:
            for b in r.boxes:
                dets.append({'c':m.names[int(b.cls[0])],'p':round(float(b.conf[0]),3),'xyxy':[round(float(v),1) for v in b.xyxy[0].tolist()]})
        print(im, json.dumps(dets[:10]))
