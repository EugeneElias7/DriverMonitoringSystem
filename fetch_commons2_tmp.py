import requests, json, re
from pathlib import Path
OUT=Path('test_assets/raw'); OUT.mkdir(parents=True,exist_ok=True)
queries=['driver behind wheel portrait','woman driver car portrait','man looking side portrait photograph','yawning woman photograph','man smoking cigarette color photograph','near infrared portrait human','driver fatigue','sleeping driver truck','person holding cigarette near face']
meta=[]
for q in queries:
    p={'action':'query','generator':'search','gsrsearch':q+' filetype:bitmap','gsrnamespace':6,'gsrlimit':6,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':1280,'format':'json'}
    r=requests.get('https://commons.wikimedia.org/w/api.php',params=p,timeout=30,headers={'User-Agent':'DMS-research/1.0'})
    print('\n###',q,r.status_code)
    for pid,pg in r.json().get('query',{}).get('pages',{}).items():
        ii=(pg.get('imageinfo') or [{}])[0]; md=ii.get('extmetadata',{})
        title=pg.get('title'); lic=md.get('LicenseShortName',{}).get('value')
        print(title,'|',lic)
        url=ii.get('thumburl') or ii.get('url')
        if url and not any(ch in url.lower() for ch in ['.pdf','.djvu','.tif','.webm','.ogv']):
            fn=OUT/(re.sub(r'[^A-Za-z0-9._-]+','_',title.replace('File:',''))[:80]);
            if not fn.suffix: fn=fn.with_suffix('.jpg')
            try:
                rr=requests.get(url,timeout=60,headers={'User-Agent':'DMS-research/1.0'}); fn.write_bytes(rr.content)
                meta.append({'query':q,'title':title,'license':lic,'url':url,'file':str(fn)})
            except Exception as e: print('dl err',e)
Path('test_assets/commons_metadata2.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print('downloaded',len(meta))
