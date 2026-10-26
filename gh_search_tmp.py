import requests, json
queries=['cigarette detection pretrained weights github','smoking detection yolov8 weights github','driver drowsiness detection pretrained model github','mediapipe face landmarker eye blink blendshapes','6drepnet head pose onnx github','nanodet plus pretrained onnx','rt-detr lightweight pretrained']
for q in queries:
    try:
        r=requests.get('https://api.github.com/search/repositories',params={'q':q,'sort':'stars','order':'desc','per_page':6},timeout=30,headers={'Accept':'application/vnd.github+json'})
        print('\n###',q,r.status_code)
        for x in r.json().get('items',[]):
            print(json.dumps({'full_name':x['full_name'],'stars':x['stargazers_count'],'updated':x['updated_at'],'license':(x.get('license') or {}).get('spdx_id'),'desc':(x.get('description') or '')[:150]},ensure_ascii=False))
    except Exception as e: print(q,'ERR',e)
