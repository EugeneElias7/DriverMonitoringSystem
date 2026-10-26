import os, glob, collections, cv2, numpy as np

p = r'C:\Users\eugen\Downloads\Driver Risk Behavior Dataset for Embedded Vision A\Driver Risk Behavior Dataset for Embedded Vision A\Images_Labels-20250704T080844Z-1-001\Images_Labels'
txts = sorted(glob.glob(os.path.join(p, '*.txt')))
counts = collections.Counter()
per = []
for t in txts:
    rows = []
    for line in open(t):
        s = line.split()
        if len(s) >= 5:
            rows.append((int(s[0]), *map(float, s[1:5])))
    if rows:
        for r in rows:
            counts[r[0]] += 1
        per.append((max(rows, key=lambda r: r[3] * r[4]), t, rows))
print('class counts', counts)
print('images with labels', len(per))
missing_image = [t for t in txts if not os.path.exists(t[:-4] + '.jpg')]
print('labels missing jpg', missing_image[:10], len(missing_image))
missing_label = [f for f in glob.glob(os.path.join(p, '*.jpg')) if not os.path.exists(f[:-4] + '.txt')]
print('jpgs missing labels', missing_label[:10], len(missing_label))

samples = []
for c in sorted(counts):
    cand = [x for x in per if any(r[0] == c for r in x[2])]
    cand.sort(key=lambda x: max(r[3] * r[4] for r in x[2] if r[0] == c), reverse=True)
    for x in cand[:2]:
        samples.append((c, x[1], x[2]))

thumbs = []
for c, t, rows in samples:
    im = cv2.imread(t[:-4] + '.jpg')
    if im is None:
        continue
    h, w = im.shape[:2]
    for r in rows:
        _, cx, cy, bw, bh = r
        x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        col = (0, 255, 0) if r[0] == c else (120, 120, 120)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 4)
        cv2.putText(im, str(r[0]), (x1 + 4, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 3)
    im = cv2.resize(im, (480, 270))
    cv2.putText(im, f'c{c} {os.path.basename(t)}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    thumbs.append(im)
while len(thumbs) % 4:
    thumbs.append(np.zeros((270, 480, 3), np.uint8))
rowsimg = [np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]
sheet = np.vstack(rowsimg)
os.makedirs(r'C:\D drive\DMS-Model-Testing\results\dataset', exist_ok=True)
cv2.imwrite(r'C:\D drive\DMS-Model-Testing\results\dataset\contact_classes.jpg', sheet)
print('sheet', sheet.shape)
