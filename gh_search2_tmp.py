import requests,json
queries=['cigarette detection','smoking detection yolo','driver drowsiness detection','driver distraction detection','head pose estimation']
for q in queries:
    r=requests.get('https://api.github.com/search/repositories',params={'q':q,'sort':'stars','order':'desc','per_page':8},timeout=30,headers={'Accept':'application/vnd.github+json','User-Agent':'dms-research'})
    print('\n###',q,r.status_code)
    d=r.json()
    if 'items' not in d: print(json.dumps(d)[:500]); continue
    for x in d['items']:
        print(json.dumps({'full_name':x['full_name'],'stars':x['stargazers_count'],'updated':x['updated_at'],'license':(x.get('license') or {}).get('spdx_id'),'desc':(x.get('description') or '')[:160]},ensure_ascii=True))
