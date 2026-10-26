import requests
for mid in ['litert-community/6DRepNet-HeadPose-LiteRT','koreashin/Driver_monitoring','Enos-123/smoking-detection']:
    print('\n########',mid)
    t=requests.get(f'https://huggingface.co/{mid}/raw/main/README.md',timeout=30).text
    print(t[:4500])
