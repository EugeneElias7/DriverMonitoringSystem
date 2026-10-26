import tensorflow as tf, numpy as np
p='models/_downloads/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
i=tf.lite.Interpreter(model_path=p); i.allocate_tensors()
print('INPUTS'); [print(d['name'],d['shape'],d['dtype']) for d in i.get_input_details()]
print('OUTPUTS'); [print(d['name'],d['shape'],d['dtype']) for d in i.get_output_details()]
print('signatures',i.get_signature_list())
# strings scan
b=open(p,'rb').read()
import re
ss=[m.group().decode('ascii','ignore') for m in re.finditer(rb'[ -~]{4,}',b)]
print([s for s in ss if s.lower() in ('person','cell phone','bicycle','car') or 'label' in s.lower()][:20])
