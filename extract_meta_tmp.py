import zipfile, io, re
b=open('models/_downloads/lite-model_efficientdet_lite0_detection_metadata_1.tflite','rb').read()
starts=[m.start() for m in re.finditer(b'PK\x03\x04',b)]
print('zip offsets',starts[:5])
for s in starts:
    try:
        z=zipfile.ZipFile(io.BytesIO(b[s:]))
        print('zip ok',z.namelist())
        for n in z.namelist():
            if 'label' in n.lower():
                txt=z.read(n).decode('utf-8','replace'); print(txt[:5000]); open('models/_downloads/efficientdet_labels.txt','w').write(txt)
        break
    except Exception as e: pass
# extract SSD V1 zip
import zipfile
z=zipfile.ZipFile('models/_downloads/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip')
print('ssd zip',z.namelist()); z.extractall('models/_downloads/ssd_v1')
