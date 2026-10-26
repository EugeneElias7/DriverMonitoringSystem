import requests, json, re
from pathlib import Path
OUT=Path('test_assets/raw'); OUT.mkdir(parents=True,exist_ok=True)
titles=['File:Portrait of young man, yawning (4419926703).jpg','File:Smoking, cigarette, portrait, man Fortepan 57490.jpg','File:Woman with hand-rolled cigarette.jpg','File:Portrait, man, smoking, cigarette Fortepan 11621.jpg','File:Infrared portrait comparison.jpg','File:Henry Fitz Jr. The Eyes Closed.jpg','File:France Nuyen-William Holden in Satan Never Sleeps trailer.jpg','File:Smoking man.jpg']
for title in titles:
    p={'action':'query','titles':title,'prop':'imageinfo','iiprop':'url|extmetadata|size','iiurlwidth':1280,'format':'json'}
    r=requests.get('https://commons.wikimedia.org/w/api.php',params=p,timeout=30,headers={'User-Agent':'DMS-research/1.0 (local testing; contact example@example.com)'})
    pages=r.json().get('query',{}).get('pages',{})
    for pid,pg in pages.items():
        ii=(pg.get('imageinfo') or [{}])[0]; md=ii.get('extmetadata',{})
        print(title,'->',pg.get('title'),md.get('LicenseShortName',{}).get('value'),ii.get('thumburl'))
        url=ii.get('thumburl') or ii.get('url')
        if url:
            fn=OUT/(re.sub(r'[^A-Za-z0-9._-]+','_',title.replace('File:',''))[:80]);
            if not fn.suffix: fn=fn.with_suffix('.jpg')
            rr=requests.get(url,timeout=90,headers={'User-Agent':'DMS-research/1.0'}); print(' dl',rr.status_code,len(rr.content)); fn.write_bytes(rr.content)
