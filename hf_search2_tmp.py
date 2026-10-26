import requests,json
terms=['yolov8n smoking','yolo11n smoking','yolov8n cigarette','yolov11n cigarette','cigarette detector','smoking detector','driver distraction','distracted driver','eye closure','blink detection']
for term in terms:
    r=requests.get('https://huggingface.co/api/models',params={'search':term,'limit':25,'full':'true','sort':'downloads','direction':-1},timeout=30)
    print('\n###',term,r.status_code)
    for m in r.json():
        files=[s['rfilename'] for s in m.get('siblings',[]) if s['rfilename'].lower().endswith(('.pt','.pth','.onnx','.tflite','.safetensors','.h5'))]
        if files:
            print(json.dumps({'id':m['id'],'dl':m.get('downloads'),'license':m.get('cardData',{}).get('license'),'files':files[:5]},ensure_ascii=True))
