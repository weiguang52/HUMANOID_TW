import json,hashlib
from pathlib import Path
import numpy as np
manifest=Path('/root/gpufree-data/datasets/practice9/tw123_v1/contact/splits/train.json')
urdf=Path('/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf')
rows=json.loads(manifest.read_text())['motions']
# Deterministic training-only audit; no validation-driven scale selection.
chosen=[rows[i] for i in np.linspace(0,len(rows)-1,64,dtype=int)]
legs=[];arms=[];speeds=[]
for r in chosen:
 with np.load(r['file']) as a:
  names=a['body_names'].tolist();p=a['body_pos_w']
  for side in ('left','right'):
   def lengths(chain):
    return sum(np.linalg.norm(p[:,names.index(x)]-p[:,names.index(y)],axis=-1)
      for x,y in zip(chain,chain[1:]))
   legs.extend(lengths([side+'_hip_linkage',side+'_mid_leg',side+'_ankle']).tolist())
   arms.extend(lengths([side+'_upper_arm',side+'_force_arm',side+'_wrist']).tolist())
  if r['category']=='walking':
   speeds.extend(np.linalg.norm(a['body_lin_vel_w'][:,names.index('base_link'),:2],axis=-1).tolist())
result=dict(urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),sample_ids=[r['id'] for r in chosen],
 method='64 deterministic training clips; FK link-origin hip-knee-ankle and shoulder-elbow-wrist chain lengths, excluding foot/hand mesh extents',
 leg_length_m=float(np.median(legs)),arm_length_m=float(np.median(arms)),
 walking_speed_quantiles_m_s=np.quantile(speeds,[.1,.5,.9,.99]).tolist())
Path('validation_artifacts/tw141_setup/scale_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sample_ids'},indent=2))
