"""Read-only reference FK, sole heights and COM acceleration; no stability gate."""
import argparse, hashlib, itertools, json
from pathlib import Path
from datetime import datetime, timezone
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads(a.plan.read_text())
asset_hash=hashlib.sha256(a.asset.read_bytes()).hexdigest()
assert asset_hash=='c42fac383f507a88c81787224e06aede10301fff18596efc12a65af584e22d5a'
m=mujoco.MjModel.from_xml_path(str(a.asset));d=mujoco.MjData(m);base=m.body('base_link').id
geoms=[m.geom(s+'_foot_collision').id for s in ['left','right']]
signs=np.array(list(itertools.product([-1,1],repeat=3)));rows=[]
for item in plan['motions']:
 f=Path(item['file']);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
 with np.load(f,allow_pickle=False) as z:
  names=z['joint_names'].tolist();addresses=[int(m.joint(n).qposadr[0]) for n in names];fps=float(z['fps'].item());assert fps==50
  com=[];heights=[];margins=[];fk=[]
  for i in range(len(z['joint_pos'])):
   d.qpos[:3]=z['root_pos'][i];d.qpos[3:7]=z['root_quat_wxyz'][i];d.qpos[addresses]=z['joint_pos'][i]
   mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);c=d.subtree_com[base].copy();com.append(c)
   corners=[(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g in geoms]
   heights.append([float(x[:,2].min()) for x in corners]);points=np.concatenate(corners)[:,:2];h=ConvexHull(points)
   margins.append(float(np.min(-(h.equations[:,:2]@c[:2]+h.equations[:,2])/np.linalg.norm(h.equations[:,:2],axis=1))))
   ids=[m.body(n).id for n in z['body_names'].tolist()];fk.append(float(np.max(np.linalg.norm(d.xpos[ids]-z['body_pos_w'][i],axis=1))))
  com=np.array(com);acc=np.diff(com,n=2,axis=0)*fps**2;heights=np.array(heights)
  assert np.isfinite(com).all() and np.isfinite(acc).all() and max(fk)<1e-5
  rows.append(dict(id=item['id'],split=item['split'],source_family=item['source_family'],reference_sha256=item['sha256'],frames=len(com),fk_max_m=max(fk),both_feet_envelope_com_outside_frames=int(np.sum(np.array(margins)<0)),envelope_margin_min_m=min(margins),sole_lowest_corner_height_min_m=heights.min(axis=0).tolist(),sole_lowest_corner_height_max_m=heights.max(axis=0).tolist(),com_acceleration_abs_p95_m_s2=np.quantile(abs(acc),.95,axis=0).tolist(),com_acceleration_abs_max_m_s2=abs(acc).max(axis=0).tolist()))
report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),asset_sha256=asset_hash,scope='All frames of unchanged eight source-isolated motions. Both-foot full-box XY envelope is optimistic geometry, NOT a loaded support polygon or dynamic stability test. Finite differences amplify reference noise. No ground alignment, filtering, reward or gate changes.',rows=rows)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows))
