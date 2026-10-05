from pathlib import Path
import json,hashlib
import numpy as np
root=Path('/root/gpufree-data/datasets/practice9');state=root/'tw154_v1';out=state/'validation';out.mkdir(exist_ok=True)
flat=['000832','001191','002011','005309','006634','007481','009516','011084']
classes={cid:'flat_forward' for cid in flat}
classes.update({'000016':'standing_upper','000124':'standing_upper','000346':'standing_upper','000039':'treadmill','000139':'nonplanar_review','000211':'turning'})
ids=list(classes);d=json.load(open(root/'tw123_v1/contact/splits/val.json'))
train=json.load(open(state/'train.json'))['motions'];families={r['source_family'] for r in train}
spec=d['contact_reference'];p=Path(spec['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
f=np.load(p);a={k:f[k] for k in f.files};offsets=np.r_[0,np.cumsum(a['lengths'])];allids=a['clip_ids'].tolist()
rows=[];bundles=[]
for cid in ids:
 i=allids.index(cid);r=dict(d['motions'][i],evaluation_class=classes[cid]);assert r['source_family'] not in families
 rows.append(r);b=a.copy()
 for k in ('contact','known','source_contact4'):b[k]=a[k][offsets[i]:offsets[i+1]]
 b['clip_ids']=np.array([cid]);b['lengths']=np.array([r['frames']]);bundles.append(b)
 dest=state/'contact/eval';dest.mkdir(parents=True,exist_ok=True);cp=dest/f'{cid}.contacts.npz';np.savez_compressed(cp,**b)
 one=dict(d,motions=[r],contact_reference=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 (dest/f'{cid}.json').write_text(json.dumps(one,indent=2))
b=bundles[0].copy()
for k in ('contact','known','source_contact4','clip_ids','lengths'):b[k]=np.concatenate([x[k] for x in bundles])
cp=out/'manifest.contacts.npz';np.savez_compressed(cp,**b)
d['motions']=rows;d['contact_reference']=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(d,indent=2))
for v in ('path','rate'):
 dst=out/v;dst.mkdir(exist_ok=True)
 for seed in (42,123,2026):
  jobs=[dict(motion_id=i,steps=r['frames']+1,output=str(dst/f'{r["id"]}.seed{seed}.json')) for i,r in enumerate(rows)]
  (dst/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
print('14 stratified validation motions; nonplanar is diagnostic, not flat acceptance')
