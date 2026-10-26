from ultralytics import YOLO
from pathlib import Path
for p in sorted(Path('models/candidates/smoking').glob('*')):
    try:
        m=YOLO(str(p))
        print('\n###',p.name,'task',m.task,'names',m.names)
        print('params',sum(x.numel() for x in m.model.parameters()) if hasattr(m.model,'parameters') else 'n/a')
        if hasattr(m,'model') and hasattr(m.model,'yaml'): print('yaml',m.model.yaml.get('scale'),m.model.yaml.get('nc'))
    except Exception as e: print('\n###',p.name,'ERR',type(e).__name__,e)
