import requests, json
ids=['MeetCool/yolov8n-smoking-detection','Beehzod/smoke_cigarette-detection2-yolo11m','Beehzod/best_smoke_cigarette-detection','Beehzod/smoking-detection-finetuned','Enos-123/smoking-detection','Notacodinggeek/yolov8n-smoking-person','cadilak/smoking-detection-yolov8','PRANAYxPATIL/smoking-detection-onnx','mosesb/drowsiness-detection-yolo-cls','ckcl/driver-drowsiness-detector','chbh7051/driver-drowsiness-detection','litert-community/6DRepNet-HeadPose-LiteRT','koreashin/Driver_monitoring','Tonjk/smoking-rfdetr-small-rtx4090']
for mid in ids:
    print('\n########',mid)
    try:
        r=requests.get(f'https://huggingface.co/api/models/{mid}',timeout=30)
        d=r.json()
        print('license:',d.get('cardData',{}).get('license'),'tags:',d.get('tags'))
        print('files:',[s['rfilename'] for s in d.get('siblings',[])])
        print('cardData:',json.dumps(d.get('cardData',{}),ensure_ascii=False)[:500])
    except Exception as e: print('ERR',e)
    try:
        c=requests.get(f'https://huggingface.co/{mid}/raw/main/README.md',timeout=30).text
        print('---README---')
        print(c[:2500])
    except Exception as e: print('README ERR',e)
