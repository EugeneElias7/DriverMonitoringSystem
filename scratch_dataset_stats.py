import os, glob, collections, cv2, numpy as np

p = r'C:\Users\eugen\Downloads\Driver Risk Behavior Dataset for Embedded Vision A\Driver Risk Behavior Dataset for Embedded Vision A\Images_Labels-20250704T080844Z-1-001\Images_Labels'
txts = sorted(glob.glob(os.path.join(p, '*.txt')))
if os.path.exists(os.path.join(p, 'classes.txt')):
    txts.remove(os.path.join(p, 'classes.txt'))
classes = ['neutral', 'microsueno', 'distraccion', 'uso_celular']
prefixes = collections.Counter()
per_class = collections.Counter()
multi = collections.Counter()
rows_per = collections.Counter()
seqs = collections.defaultdict(collections.Counter)
for t in txts:
    rows = [line.split() for line in open(t) if len(line.split()) >= 5]
    rows_per[len(rows)] += 1
    ids = tuple(sorted(set(int(r[0]) for r in rows)))
    multi[ids] += 1
    for i in ids:
        per_class[i] += 1
    stem = os.path.basename(t)
    prefix = stem.rsplit('_', 1)[0]
    prefixes[prefix] += 1
    for i in ids:
        seqs[prefix][i] += 1
print('classes:', classes)
print('images:', len(txts))
print('images per class:', {classes[k]: v for k, v in sorted(per_class.items())})
print('multi-label combos:', {tuple(classes[i] for i in k): v for k, v in multi.most_common()})
print('rows per image:', dict(rows_per))
print('unique sequences/prefixes:', len(prefixes))
for pfx, n in prefixes.most_common(20):
    print(pfx, n, {classes[k]: v for k, v in sorted(seqs[pfx].items())})
