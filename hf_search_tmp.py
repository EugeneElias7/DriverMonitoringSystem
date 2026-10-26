import requests, json
for term in ['cigarette','smoking detection','smoking','drowsiness','driver monitoring','head pose','eye state','yolo smoking']:
    try:
        r=requests.get('https://huggingface.co/api/models',params={'search':term,'limit':40,'full':'true'},timeout=30)
        print('\n###',term,r.status_code)
        for m in r.json():
            files=[s['rfilename'] for s in m.get('siblings',[]) if s['rfilename'].lower().endswith(('.pt','.pth','.onnx','.tflite','.bin','.safetensors','.joblib','.pkl'))]
            print(json.dumps({'id':m['id'],'downloads':m.get('downloads'),'likes':m.get('likes'),'tags':[t for t in m.get('tags',[]) if t in ('yolov8','yolov5','yolo11','object-detection','image-classification','tflite','onnx','license:agpl-3.0','license:mit','license:apache-2.0')],'files':files[:8],'license':m.get('cardData',{}).get('license')},ensure_ascii=False))
    except Exception as e: print(term,'ERR',e)
