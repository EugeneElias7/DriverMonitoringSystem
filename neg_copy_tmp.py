from pathlib import Path
from PIL import Image
out=Path('test_assets/curated')
for a,b in [('driver_frontal.jpg','smoking_negative_1.jpg'),('face_open.jpg','smoking_negative_2.jpg'),('face_yawning.jpg','smoking_negative_3.jpg'),('smoking_cigarette_in_hand.jpg','smoking_hard_1.jpg')]:
    Image.open(out/a).convert('RGB').save(out/b,quality=92)
    print(b)
