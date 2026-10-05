from pathlib import Path
import json,hashlib,shutil
import numpy as np
root=Path('/root/gpufree-data/datasets/practice9')
state=root/'tw151_v1';out=state/'validation';out.mkdir(exist_ok=True)
ids=['000832','001191','002011','005309','006634','007481','009516','011084']
audit={r['id']:r for r in json.load(open(state/'source_audit.json'))}
d=json.load(open(root/'tw123_v1/contact/splits/val.json'))
train=json.load(open(root/'tw123_v1/contact/splits/train.json'))['motions']
tids={r['id'] for r in train};families={r['source_family'] for r in train}
spec=d['contact_reference'];p=Path(spec['file'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256']
a=np.load(p);offsets=np.r_[0,np.cumsum(a['lengths'])];all_ids=a['clip_ids'].tolist()
assert all_ids==[r['id'] for r in d['motions']]
rows=[];bundles=[]
for cid in ids:
 i=all_ids.index(cid);r=d['motions'][i]
 assert cid not in tids and r['source_family'] not in families
 assert audit[cid]['straightness']>.9 and audit[cid]['low_foot_p95_p05_m']<.05
 r=dict(r,evaluation_class='flat_forward_candidate')
 rows.append(r)
 b={k:a[k].copy() for k in a.files}
 for k in ('contact','known','source_contact4'):b[k]=a[k][offsets[i]:offsets[i+1]]
 b['clip_ids']=np.array([cid]);b['lengths']=np.array([r['frames']]);bundles.append(b)
 target=state/'contact/eval';target.mkdir(parents=True,exist_ok=True)
 cp=target/f'{cid}.contacts.npz';np.savez_compressed(cp,**b)
 one=dict(d,motions=[r],contact_reference=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 (target/f'{cid}.json').write_text(json.dumps(one,indent=2))
b=bundles[0].copy()
for k in ('contact','known','source_contact4','clip_ids','lengths'):b[k]=np.concatenate([x[k] for x in bundles])
cp=out/'manifest.contacts.npz';np.savez_compressed(cp,**b)
d['motions']=rows;d['contact_reference']=dict(spec,file=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(d,indent=2))
(state/'selection.json').write_text(json.dumps([audit[cid] for cid in ids],indent=2))
for v in ('size','residual'):
 dst=out/v;dst.mkdir(exist_ok=True);(state/v).mkdir(exist_ok=True)
 for name in ('final_checkpoint','source_commit','data_audit.json'):
  shutil.copy2(root/'tw141_v1'/v/name,state/v/name)
 for seed in (42,123,2026):
  jobs=[dict(motion_id=i,steps=r['frames']+1,output=str(dst/f'{r["id"]}.seed{seed}.json')) for i,r in enumerate(rows)]
  (dst/f'jobs{seed}.json').write_text(json.dumps(jobs,indent=2))
print('Prepared 8 source-disjoint flat-forward candidates; 24 evaluations per policy')
