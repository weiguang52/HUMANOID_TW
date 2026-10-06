from pathlib import Path
import sys,json,importlib.util,hashlib
import numpy as np,torch
from contact_labels import from_features,resample
root=Path('/root/gpufree-data/datasets/practice9');out=Path('validation_artifacts/tw176_setup')
urdf=root/'custom_robot/urdf/urdf0711_training_30dof.urdf'
spec=importlib.util.spec_from_file_location('sole','source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/sole_tracking.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
vertices,_=s.read_foot_geometry(urdf);corners=torch.tensor(vertices,dtype=torch.float32)
m=json.loads((root/'tw154_v1/train.json').read_text());rows=[]
human=root/'humanml3d_rebuild/staging-v1/HumanML3D'
for r in m['motions']:
 if r['category'] not in ['walk','walking']:continue
 j=np.load(human/'new_joints'/f"{r['id']}.npy");f=np.load(human/'new_joint_vecs'/f"{r['id']}.npy")
 c,k,h,raw=from_features(f,j);rc,rk=resample(c,k,r['frames'])
 a=np.load(r['file']);bn=a['body_names'].tolist();ix=[bn.index(n) for n in ['left_foot','right_foot']]
 pos=torch.from_numpy(a['body_pos_w'][:,ix]).float();q=torch.from_numpy(a['body_quat_w'][:,ix]).float()
 z=s.sole_height(pos,q,corners).numpy()
 rows.append(dict(id=r['id'],raw_swing=float((c==0).mean()),raw_swing_h_ge_5cm=float(((c==0)&(h>=.05)).mean()),
  known_source_swing=float(((c==0)&k.astype(bool)).mean()),known_resampled_swing=float(((rc==0)&rk.astype(bool)).mean()),
  reference_sole_min_m=float(z.min()),reference_sole_p01_m=float(np.quantile(z,.01)),reference_sole_max_m=float(z.max())))
keys=[k for k in rows[0] if k!='id'];summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
result=dict(rows=rows,summary=summary,urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
 note='Walking training clips only, equal-clip means; source human geometry thresholds, no label/mask mutation; minimum collision-box Z in FK world, not contact-force truth.')
(out/'mask_geometry_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
