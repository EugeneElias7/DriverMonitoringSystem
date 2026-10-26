import requests, json
for pkg in ['mediapipe','ai-edge-litert','onnxruntime','tflite-runtime','tensorflow']:
    r=requests.get(f'https://pypi.org/pypi/{pkg}/json',timeout=30)
    if r.status_code!=200: print(pkg,r.status_code); continue
    d=r.json(); ver=d['info']['version']
    files=[(f['filename'],round(f['size']/1e6,1)) for f in d['releases'].get(ver,[]) if 'aarch64' in f['filename'] or 'armv7' in f['filename']]
    print(pkg,ver,'aarch64/arm wheels:',files[:6])
