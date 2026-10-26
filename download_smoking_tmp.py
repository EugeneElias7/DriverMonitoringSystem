import requests, hashlib
from pathlib import Path
out=Path('models/candidates/smoking'); out.mkdir(parents=True,exist_ok=True)
files={
 'beehzod_yolo11m_cigarette2.pt':'https://huggingface.co/Beehzod/smoke_cigarette-detection2-yolo11m/resolve/main/best.pt',
 'beehzod_yolo11m_finetuned.pt':'https://huggingface.co/Beehzod/smoking-detection-finetuned/resolve/main/best.pt',
 'meetcool_yolov8n.pt':'https://huggingface.co/MeetCool/yolov8n-smoking-detection/resolve/main/models/fine_tune_YOLOv8n.pt',
 'meetcool_yolov8n.onnx':'https://huggingface.co/MeetCool/yolov8n-smoking-detection/resolve/main/models/fine_tune_YOLOv8n.onnx',
 'notacodinggeek_yolov8n.pt':'https://huggingface.co/Notacodinggeek/yolov8n-smoking-person/resolve/main/best.pt',
 'cadilak_yolov8_2class.pt':'https://huggingface.co/cadilak/smoking-detection-yolov8/resolve/main/best_smoking_detect_2class.pt',
 'abhishek_yolo_smoking.pt':'https://raw.githubusercontent.com/abhishekkrjha2811/YOLO-Based_Smoking_Activity_Detector/main/best.pt',
 'leviathan_cgr.onnx':'https://raw.githubusercontent.com/Leviathanlzx/cgr_detection/main/models/last.onnx',
 'leviathan_pose.onnx':'https://raw.githubusercontent.com/Leviathanlzx/cgr_detection/main/models/yolov8n-pose.onnx',
}
for name,url in files.items():
    p=out/name
    if p.exists(): print('exists',name); continue
    try:
        r=requests.get(url,timeout=300,headers={'User-Agent':'dms-research'})
        print(name,r.status_code,len(r.content),r.headers.get('content-type'))
        if r.ok and len(r.content)>100000:
            p.write_bytes(r.content); print(' sha256',hashlib.sha256(r.content).hexdigest())
        else:
            print(' body head',r.text[:120])
    except Exception as e: print(name,'ERR',e)
