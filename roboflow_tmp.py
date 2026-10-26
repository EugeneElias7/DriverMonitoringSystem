import requests
urls=['https://universe.roboflow.com/smoking-detection-5e8hh/smoking-detection-ggnhq','https://universe.roboflow.com/richie-lab/smoking-tasfx','https://universe.roboflow.com/yolo-pdvpx/cigarette-h2p1m','https://universe.roboflow.com/xu-wei/cigarette-all']
for u in urls:
    try:
        r=requests.get(u,timeout=30,headers={'User-Agent':'Mozilla/5.0'})
        print('\n###',u,r.status_code,len(r.text))
        low=r.text.lower()
        for key in ['license','cc by','mit','apache','non-commercial','class','cigarette']:
            idx=low.find(key)
            if idx>=0: print(key,repr(r.text[max(0,idx-150):idx+250])[:500])
    except Exception as e: print('ERR',e)
