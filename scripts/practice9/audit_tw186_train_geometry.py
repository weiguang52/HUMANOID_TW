import sys,json,pathlib,numpy as np,torch
sys.path.insert(0,'scripts/practice9')
from diagnose_sole_contacts import g,URDF
from contact_labels import from_features,resample
root=pathlib.Path('/root/gpufree-data/datasets/practice9');human=root/'humanml3d_rebuild/staging-v1/HumanML3D';m=json.load(open(root/'tw154_v1/train.json'));corners=torch.tensor(g.read_foot_geometry(URDF)[0]).float();rows=[]
for r in m['motions']:
 if r['category'] not in ['walk','walking']:continue
 c,k,_,_=from_features(np.load(human/'new_joint_vecs'/f"{r['id']}.npy"),np.load(human/'new_joints'/f"{r['id']}.npy"));c,k=resample(c,k,r['frames']);a=np.load(r['file']);bn=a['body_names'].tolist();ix=[bn.index(n) for n in ['left_foot','right_foot']];z=g.sole_height(torch.from_numpy(a['body_pos_w'][:,ix]).float(),torch.from_numpy(a['body_quat_w'][:,ix]).float(),corners).numpy();mask=k.astype(bool)&c.astype(bool)
 rows.append(dict(id=r['id'],stance_samples=int(mask.sum()),stance_above_5mm=float((z[mask]>.005).mean()) if mask.any() else None,stance_above_10mm=float((z[mask]>.01).mean()) if mask.any() else None,stance_height_m=float(z[mask].mean()) if mask.any() else None))
out=pathlib.Path('validation_artifacts/tw186_contact_audit/train_geometry.json');out.write_text(json.dumps(rows,indent=2));print(len(rows),{key:float(np.mean([r[key] for r in rows if r[key] is not None])) for key in ['stance_above_5mm','stance_above_10mm','stance_height_m']})
