import requests
from pathlib import Path
titles=['File:Portrait of young man, yawning (4419926703).jpg','File:Smoking, cigarette, portrait, man Fortepan 57490.jpg','File:Henry Fitz Jr. The Eyes Closed.jpg','File:France Nuyen-William Holden in Satan Never Sleeps trailer.jpg','File:Series of three photos showing a hysterical yawning woman Wellcome L0034940 (cropped).jpg','File:Fashion Time, Modeling Photography in Tbilisi - Iranian Model - Jorj Barber 22.jpg','File:Fashion Time, Modeling Photography in Tbilisi - Iranian Model - Jorj Barber 23.jpg','File:Infrared portrait comparison.jpg']
for t in titles:
    name=t.replace('File:','')
    url='https://commons.wikimedia.org/wiki/Special:FilePath/'+requests.utils.quote(name)+'?width=1000'
    r=requests.get(url,timeout=90,headers={'User-Agent':'DMS-research/1.0 (local testing)'})
    print(t,r.status_code,len(r.content),r.headers.get('content-type'))
    if r.ok and 'image' in r.headers.get('content-type',''):
        fn=Path('test_assets/raw')/(requests.utils.quote(name,safe='')[:70]+'.jpg')
        fn.write_bytes(r.content); print(' ->',fn)
