"""Merge unchanged contact sidecars for six mixed-category holdout clips."""
from pathlib import Path
import json, hashlib
import numpy as np
root=Path('/root/gpufree-data/datasets/practice9')
out=root/'tw141_v1/validation'
out.mkdir(parents=True,exist_ok=True)
ids=['000016','000124','000346','000039','000139','000211']
rows=[]; arrays=[]; template=None
train=json.loads((root/'tw123_v1/contact/splits/train.json').read_text())
train_ids={r['id'] for r in train['motions']}
train_families={r.get('source_family') for r in train['motions']}
for clip in ids:
    d=json.loads((root/f'tw123_v1/contact/eval/{clip}.json').read_text())
    r=d['motions'][0]
    assert r['id']==clip and clip not in train_ids and r['split']=='val'
    assert r['source_family'] not in train_families
    assert r['category'] in ('standing_upper','walking')
    spec=d['contact_reference'];p=Path(spec['file'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
    with np.load(p) as a:
        assert a['clip_ids'].tolist()==[clip] and a['lengths'].tolist()==[r['frames']]
        arrays.append({k:a[k].copy() for k in a.files})
    rows.append(r);template=d
bundle=out/'manifest.contacts.npz'
a=arrays[0].copy()
for k in ('contact','known','source_contact4','clip_ids','lengths'):
    a[k]=np.concatenate([x[k] for x in arrays])
np.savez_compressed(bundle,**a)
template['motions']=rows
template['contact_reference']=dict(template['contact_reference'],file=str(bundle),sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(template,indent=2))
for variant in ('size','residual'):
    dst=out/variant;dst.mkdir(exist_ok=True)
    for seed in (42,123,2026):
        jobs=[dict(motion_id=i,steps=max(1000,r['frames']+1),output=str(dst/f'{r["id"]}.seed{seed}.json')) for i,r in enumerate(rows)]
        (dst/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
print([(r['id'],r['frames']) for r in rows])
