import requests
for repo in ['ultralytics/ultralytics','ultralytics/yolov5','RangiLyu/nanodet','PaddlePaddle/PaddleDetection','thohemp/6DRepNet','TadasBaltrusaitis/OpenFace']:
    r=requests.get(f'https://api.github.com/repos/{repo}',timeout=30,headers={'User-Agent':'dms-research'})
    d=r.json(); print(repo,'|',(d.get('license') or {}).get('spdx_id'),'|',(d.get('license') or {}).get('name'),'| stars',d.get('stargazers_count'))
    lic=requests.get(f'https://raw.githubusercontent.com/{repo}/master/LICENSE',timeout=30)
    if lic.status_code!=200: lic=requests.get(f'https://raw.githubusercontent.com/{repo}/main/LICENSE',timeout=30)
    print(' license head:',lic.text[:180].replace('\n',' ') if lic.ok else lic.status_code)
