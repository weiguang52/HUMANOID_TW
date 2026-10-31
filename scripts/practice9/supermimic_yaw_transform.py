"""Exact quarter-turn coordinate transforms of73D world-root robot states."""
import math
import torch
from mjlab.utils.lab_api.math import quat_mul

def rotate_world_state(state,turns):
 assert state.ndim==2 and state.shape[1]==73
 turns=int(turns)%4;out=state.clone()
 if turns==0:return out
 for start in [0,7,10]:
  x=state[:,start];y=state[:,start+1]
  if turns==1:out[:,start],out[:,start+1]=-y,x
  elif turns==2:out[:,start],out[:,start+1]=-x,-y
  else:out[:,start],out[:,start+1]=y,-x
 c,s={1:(math.sqrt(.5),math.sqrt(.5)),2:(0.,1.),3:(math.sqrt(.5),-math.sqrt(.5))}[turns]
 q=state.new_tensor([c,0,0,s]).expand(len(state),-1)
 out[:,3:7]=quat_mul(q,state[:,3:7])
 return out

def state_metrics(pred,target):
 p=pred.double();t=target.double();error=p-t
 pq=torch.nn.functional.normalize(p[:,3:7],dim=-1);tq=torch.nn.functional.normalize(t[:,3:7],dim=-1)
 tq=torch.where((pq*tq).sum(-1,keepdim=True)<0,-tq,tq)
 angle=4*torch.atan2((pq-tq).norm(dim=-1),(pq+tq).norm(dim=-1))
 error[:,3:7]=pq-tq
 return dict(root_position_p95_m=float((p[:,:3]-t[:,:3]).norm(dim=-1).quantile(.95)),root_orientation_p95_rad=float(angle.quantile(.95)),root_linear_velocity_rmse_m_s=float(error[:,7:10].square().mean().sqrt()),root_angular_velocity_rmse_rad_s=float(error[:,10:13].square().mean().sqrt()),joint_velocity_rmse_rad_s=float(error[:,43:73].square().mean().sqrt()),max_state_component_error=float(error.abs().max()))
