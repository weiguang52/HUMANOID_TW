import json,numpy as np
from pathlib import Path
root=Path('/root/gpufree-data/datasets/practice9/tw154_v1')
rows=[r for r in json.load(open(root/'train.json'))['motions'] if r['category']=='walking']
chosen=[rows[i] for i in np.linspace(0,len(rows)-1,min(128,len(rows)),dtype=int)]
result=[]
for r in chosen:
 a=np.load(r['file']);names=a['joint_names'].tolist()
 ix=[i for i,n in enumerate(names) if 'hip_pitch' in n or 'knee_pitch' in n]
 q=a['joint_pos'][:,ix]
 for scale in (.5,.75):
  target=q.copy();cap=3.*scale*.02
  for t in range(1,len(q)):target[t]=target[t-1]+np.clip(q[t]-target[t-1],-cap,cap)
  span=lambda x:np.percentile(x,95,axis=0)-np.percentile(x,5,axis=0)
  result.append(dict(id=r['id'],scale=scale,rms_error_deg=float(np.sqrt(np.mean((q-target)**2))*180/np.pi),
   amplitude_ratio=float(np.mean(span(target)/np.maximum(span(q),1e-6)))))
summary={str(scale):dict(mean_rms_error_deg=float(np.mean([r['rms_error_deg'] for r in result if r['scale']==scale])),
 mean_amplitude_ratio=float(np.mean([r['amplitude_ratio'] for r in result if r['scale']==scale]))) for scale in (.5,.75)}
(root/'rate_audit.json').write_text(json.dumps(dict(summary=summary,rows=result,note='Kinematic reference target limiter only, no dynamics or policy. 128 deterministic training clips.'),indent=2))
print(json.dumps(summary,indent=2))
