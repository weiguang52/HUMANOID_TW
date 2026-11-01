"""Independent named NumPy/SciPy audit of non-orientation goal moments."""
import argparse,hashlib,json,math
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--parent',type=Path,required=True)
p.add_argument('--mjcf',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=torch.load(a.calibration,map_location='cpu',weights_only=False)
assert z['kind']=='offline_training_goal_normalizer_calibration_not_policy_checkpoint'
assert z['parent_sha256']==sha(a.parent)
parent=torch.load(a.parent,map_location='cpu',weights_only=False)
assert sha(a.mjcf)==parent['cfg']['extra']['tw_mjcf_sha256']
model=mujoco.MjModel.from_xml_path(str(a.mjcf))
names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt) if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE]
assert len(names)==30
q0=np.array([-math.pi/2 if n=='left_shoulder_roll_joint' else math.pi/2 if n=='right_shoulder_roll_joint' else 0 for n in names],dtype=np.float32).astype(np.float64)
sums=np.zeros((10,75));sq=np.zeros_like(sums);count=0
for row in z['inputs']:
 path=Path(row['file']);assert sha(path)==row['sha256']
 with np.load(path,allow_pickle=False) as d:
  order=[d['joint_names'].tolist().index(n) for n in names]
  q=d['joint_pos'][:,order].astype(np.float64)-q0
  v=d['joint_vel'][:,order].astype(np.float64)
  anchor=d['body_names'].tolist().index('chest')
  pos=d['body_pos_w'][:,anchor].astype(np.float64)
  quat=d['body_quat_w'][:,anchor].astype(np.float64)
  rot=Rotation.from_quat(quat[:,[1,2,3,0]])
  lin=d['body_lin_vel_w'][:,anchor].astype(np.float64)
  ang=d['body_ang_vel_w'][:,anchor].astype(np.float64)
  n=len(q);assert n==row['frames'];count+=n
  for h in range(10):
   idx=np.minimum(np.arange(n)+h+1,n-1)
   goal=np.concatenate([q[idx],v[idx],rot.inv().apply(pos[idx]-pos),np.zeros((n,6)),rot[idx].inv().apply(lin[idx]),rot[idx].inv().apply(ang[idx])],axis=1)
   sums[h]+=goal.sum(0);sq[h]+=(goal*goal).sum(0)
mean=sums/count;var=sq/count-mean**2
mask=z['replacement_mask'].numpy().reshape(10,75)
assert np.array_equal(mask,np.tile(np.r_[np.ones(63,bool),np.zeros(6,bool),np.ones(6,bool)],(10,1)))
errors={'mean':float(np.abs(mean-z['goal_mean'].numpy().reshape(10,75))[mask].max()),'variance':float(np.abs(var-z['goal_var'].numpy().reshape(10,75))[mask].max())}
assert count==z['fit_frames'] and all(x<=1e-6 for x in errors.values()),errors
src=parent['actor_critic'];keep=~z['replacement_mask']
assert torch.equal(z['goal_mean'][keep],src['goal_norm.mean'][keep])
assert torch.equal(z['goal_var'][keep],src['goal_norm.var'][keep])
assert torch.equal(z['frozen_mask'],src['goal_norm.frozen_mask'])
a.output.write_text(json.dumps({'passed':True,'method':'independent named NumPy moments and SciPy quaternion frame transforms','frames':count,'dimensions_checked':int(mask.sum()),'absolute_moment_tolerance':1e-6,'max_errors':errors,'preserved_dimensions_checked':int(keep.sum()),'calibration_sha256':sha(a.calibration),'mjcf_sha256':sha(a.mjcf),'joint_names':names,'scope':'Calibration construction audit only; no training, test arrays or physical-control acceptance'},indent=2)+'\n')
print(json.dumps(errors))
