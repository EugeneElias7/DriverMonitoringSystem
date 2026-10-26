import requests,json
repos=['Leviathanlzx/cgr_detection','ravigohel142996/Cigarette-detection','RichardoMrMu/yolov5-smoke-detection-python','mehulpurohit97/Cigarette-Smoking-Detection-using-Deep-Learning','qunshansj/YOLO_Smoking_PhoneUse_Detection','abhishekkrjha2811/YOLO-Based_Smoking_Activity_Detector']
for repo in repos:
    print('\n########',repo)
    try:
        r=requests.get(f'https://api.github.com/repos/{repo}',timeout=30,headers={'User-Agent':'dms-research'})
        d=r.json(); print('license',(d.get('license') or {}).get('spdx_id'),'size KB',d.get('size'),'default',d.get('default_branch'))
        c=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{d.get("default_branch","main")}?recursive=1',timeout=30,headers={'User-Agent':'dms-research'})
        tree=c.json().get('tree',[])
        print('files', [t['path'] for t in tree if t['path'].lower().endswith(('.pt','.pth','.onnx','.tflite','.pb','.h5','.engine','.weights'))][:20])
        print('readmes',[t['path'] for t in tree if t['path'].lower().startswith(('readme','license'))][:5])
    except Exception as e: print('ERR',e)
