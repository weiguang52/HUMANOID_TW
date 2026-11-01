"""Residual bounded analytic contact-position feedback diagnostic, not a learned policy."""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import mujoco

class ResidualContactBalance:
 def __init__(self,model,joint_names):
  self.model=model;self.data=mujoco.MjData(model);self.names=list(joint_names)
  self.qadr=[int(model.joint(n).qposadr[0]) for n in self.names]
  self.lower=[i for i,n in enumerate(self.names) if n.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot'))]
  assert len(self.lower)==14
  self.jids=[model.joint(self.names[i]).id for i in self.lower]
  self.cols=[int(model.joint(self.names[i]).dofadr[0]) for i in self.lower]
  assert all(model.jnt_limited[j] for j in self.jids)
 def target(self,state,nominal,reference_quat,reference_omega,feet,enabled):
  m=self.model;d=self.data;state=np.asarray(state,dtype=float);nominal=np.asarray(nominal,dtype=float);q=state[13:43]
  lb=np.maximum(-.06,m.jnt_range[self.jids,0]-q[self.lower]);ub=np.minimum(.06,m.jnt_range[self.jids,1]-q[self.lower]);assert np.all(lb<ub),'Original state outside one-step joint bounds'
  desired=np.clip(nominal[self.lower]-q[self.lower],lb,ub);target=nominal.copy();target[self.lower]=q[self.lower]+desired
  actual=Rotation.from_quat(state[[4,5,6,3]]);reference=Rotation.from_quat(np.asarray(reference_quat)[[1,2,3,0]])
  error=(actual.inv()*reference).as_rotvec()+.02*actual.inv().apply(np.asarray(reference_omega)-state[10:13]);delta=error[:2];delta*=min(1.,np.deg2rad(1)/max(float(np.linalg.norm(delta)),1e-12))
  info=dict(enabled=bool(enabled),root_body_frame_delta_rad=delta.tolist(),loaded_points=0,solver_success=True,solver_status=None,lower_bound_rad=lb.tolist(),upper_bound_rad=ub.tolist(),baseline_increment_rad=desired.tolist())
  if enabled:
   d.qpos[:3]=state[:3];d.qpos[3:7]=state[3:7];d.qpos[self.qadr]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mat=[]
   for foot in feet.values():
    for c in foot['contacts']:
     if not c['ground'] or c['normal_force_n']<=.01:continue
     name=next(g for g in c['geoms'] if '_foot_collision' in g).replace('robot/','');body=int(m.geom_bodyid[m.geom(name).id]);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,np.asarray(c['pos_w']),body);mat.append(jp)
   info['loaded_points']=len(mat)
   if mat and float(np.linalg.norm(delta))>1e-12:
    j=np.concatenate(mat);jl=j[:,self.cols];wanted=-j[:,3:5]@delta
    # Fixed 1mm/rad tie-breaker, recorded before trials; never tune on holdout.
    regularizer=.001;matrix=np.vstack([jl,regularizer*np.eye(14)]);rhs=np.r_[wanted,np.zeros(14)]
    sol=lsq_linear(matrix,rhs,bounds=(lb-desired,ub-desired),tol=1e-12,max_iter=200)
    assert sol.success and np.isfinite(sol.x).all(),'Preserve failed solve rather than mask it'
    assert np.all(sol.x>=lb-desired-1e-10) and np.all(sol.x<=ub-desired+1e-10)
    target[self.lower]=q[self.lower]+desired+sol.x
    info.update(solver_status=int(sol.status),contact_residual_max_m=float(abs(jl@sol.x-wanted).max()),active_bound_joint_names=[self.names[self.lower[k]] for k in range(14) if sol.active_mask[k]])
  assert np.isfinite(target).all()
  return target,info
