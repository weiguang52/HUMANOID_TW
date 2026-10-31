import json,numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pathlib import Path
r=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031');m=mujoco.MjModel.from_xml_path(str(r/'robot/tw30.xml'));d=mujoco.MjData(m);d.qpos[:]=m.qpos0;d.qpos[2]=.264
for side,sgn in [('left',-1),('right',1)]:d.qpos[m.joint(side+'_shoulder_roll_joint').qposadr[0]]=sgn*np.deg2rad(10)
mujoco.mj_kinematics(m,d);feet=[m.body(s+'_foot').id for s in ['left','right']];positions=d.xpos[feet].copy();rotations=d.xmat[feet].reshape(2,3,3).copy();names=[m.joint(i).name for i in range(m.njnt) if m.joint(i).name and m.joint(i).name.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot_roll','right_foot_roll'))];adr=[m.joint(n).qposadr[0] for n in names];bounds=np.array([m.joint(n).range for n in names]);d.qpos[2]-=.002

def residual(v):
 q=v[:-1];d.qpos[0]=v[-1]
 d.qpos[adr]=q;mujoco.mj_kinematics(m,d)
 pos=(d.xpos[feet]-positions).ravel()/.01
 rot=np.concatenate([Rotation.from_matrix(d.xmat[i].reshape(3,3)@target.T).as_rotvec() for i,target in zip(feet,rotations)])/.1
 mujoco.mj_comPos(m,d)
 com=d.subtree_com[m.body('base_link').id]
 return np.r_[pos,rot,(com[:2]-np.array([.010,0]))/.01,q*.001]
for i,n in enumerate(names):
 if 'knee_pitch' in n:bounds[i,0]=0
 if 'yaw' in n:bounds[i]=np.maximum(bounds[i,0],-.0872665),np.minimum(bounds[i,1],.0872665)
seed=np.zeros(len(names))
for i,n in enumerate(names):
 if 'knee_pitch' in n:seed[i]=.2
res=least_squares(residual,np.r_[seed,0],bounds=(np.r_[bounds[:,0],-.025],np.r_[bounds[:,1],.025]),max_nfev=2000,ftol=1e-12,xtol=1e-12,gtol=1e-12);residual(res.x);err=float(np.max(np.linalg.norm(d.xpos[feet]-positions,axis=1)));angle=max(np.linalg.norm(Rotation.from_matrix(d.xmat[i].reshape(3,3)@target.T).as_rotvec()) for i,target in zip(feet,rotations));out=dict(root_z_m=float(d.qpos[2]),root_x_m=float(d.qpos[0]),joint_targets=dict(zip(names,res.x[:-1].tolist())),com_world_m=d.subtree_com[m.body('base_link').id].tolist(),max_foot_position_error_m=err,max_foot_orientation_error_rad=float(angle),solver_success=bool(res.success),geometry_pass=bool(err<1e-5 and angle<1e-4),scope='Geometry only, not dynamic stability; arm roll +/-10deg')
(r/'longterm_diagnostics/bounded_forward_crouch2mm_ik.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
