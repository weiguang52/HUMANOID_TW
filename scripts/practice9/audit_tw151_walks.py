import json,re
from pathlib import Path
import numpy as np
root=Path('/root/gpufree-data/datasets/practice9')
sel={r['id']:r for r in json.load(open(root/'tw75_dataset_v1/selection.json'))}
rows=[]
for split in ('train','val'):
 for r in json.load(open(root/f'tw123_v1/contact/splits/{split}.json'))['motions']:
  if r['category']!='walking':continue
  x=sel[r['id']];a=np.load(x['source']);p=a[:,0];f=a[:,[7,8,10,11],1].min(1)
  path=np.linalg.norm(np.diff(p[:,[0,2]],axis=0),axis=1).sum()
  span=float(np.quantile(f,.95)-np.quantile(f,.05))
  rows.append(dict(id=r['id'],split=split,source_family=x['source_family'],captions=x['captions'],
    root_height_span_m=float(np.ptp(p[:,1])),low_foot_p95_p05_m=span,
    nonplanar_review_flag=span>.15,straightness=float(np.linalg.norm(p[-1,[0,2]]-p[0,[0,2]])/max(path,1e-6)),
    source_speed_m_s=float(path/(len(a)/20)),frames=r['frames']))
out=root/'tw151_v1/source_audit.json';out.write_text(json.dumps(rows,indent=2))
for split in ('train','val'):
 rr=[r for r in rows if r['split']==split]
 print(split,len(rr),'nonplanar_review_flags',sum(r['nonplanar_review_flag'] for r in rr))
for r in rows:
 if r['split']=='val' and r['low_foot_p95_p05_m']<.05 and r['straightness']>.9 and re.search('walking_(slow|medium|fast)|WalkingStraightForwards',r['source_family']):
  print(r['id'],round(r['source_speed_m_s'],2),r['source_family'],r['captions'])
