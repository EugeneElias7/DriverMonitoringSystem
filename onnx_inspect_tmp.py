import onnxruntime as ort, numpy as np
from pathlib import Path
for p in ['leviathan_cgr.onnx','meetcool_yolov8n.onnx','leviathan_pose.onnx']:
    s=ort.InferenceSession('models/candidates/smoking/'+p,providers=['CPUExecutionProvider'])
    print('\n###',p)
    for i in s.get_inputs(): print('IN',i.name,i.shape,i.type)
    for o in s.get_outputs(): print('OUT',o.name,o.shape,o.type)
    md=s.get_modelmeta().custom_metadata_map
    print('metadata',{k:str(v)[:300] for k,v in md.items() if k in ('names','description','author','version','task','imgsz','stride')})
