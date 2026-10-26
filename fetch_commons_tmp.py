import requests, json, re
from pathlib import Path
OUT=Path('test_assets/raw'); OUT.mkdir(parents=True,exist_ok=True)
queries=['cigarette smoking man portrait','woman smoking cigarette portrait','cigarette in mouth','sleeping man face','closed eyes portrait','yawning man portrait','driver driving car','infrared portrait','night vision face','looking sideways portrait']
allmeta=[]
for q in queries:
    p={'action':'query','generator':'search','gsrsearch':q,'gsrnamespace':6,'gsrlimit':8,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':1280,'format':'json'}
    r=requests.get('https://commons.wikimedia.org/w/api.php',params=p,timeout=30,headers={'User-Agent':'DMS-research/1.0 (local testing)'})
    print('\n###',q,r.status_code)
    pages=r.json().get('query',{}).get('pages',{})
    for pid,pg in pages.items():
        ii=(pg.get('imageinfo') or [{}])[0]; md=ii.get('extmetadata',{})
        title=pg.get('title')
        lic=md.get('LicenseShortName',{}).get('value'); artist=re.sub('<[^>]+>','',md.get('Artist',{}).get('value',''))[:80]
        url=ii.get('thumburl') or ii.get('url')
        size=ii.get('size')
        print(json.dumps({'title':title,'license':lic,'artist':artist,'url':url,'size':size},ensure_ascii=True))
        if url:
            fn=OUT/(re.sub(r'[^A-Za-z0-9._-]+','_',title.replace('File:',''))[:80])
            if not fn.suffix: fn=fn.with_suffix('.jpg')
            if not fn.exists():
                try:
                    rr=requests.get(url,timeout=60,headers={'User-Agent':'DMS-research/1.0'}); fn.write_bytes(rr.content)
                except Exception as e: print('dl err',e)
            allmeta.append({'query':q,'title':title,'license':lic,'artist':artist,'url':url,'file':str(fn)})
Path('test_assets/commons_metadata.json').write_text(json.dumps(allmeta,indent=2,ensure_ascii=False),encoding='utf-8')
print('saved',len(allmeta))
