"""Explicit diagnostic world-yaw frame; no implicit checkpoint or policy loading."""
import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_mul
CONTRACT='root_xy_yaw_canonical_world_velocity_v1'
KIND='offline_yaw_canonical_dynamics_not_policy_checkpoint'

def rotate_vector(v,c,s):
 return torch.cat([c*v[:,:1]-s*v[:,1:2],s*v[:,:1]+c*v[:,1:2],v[:,2:3]],dim=1)

def canonicalize_state(state):
 assert state.ndim==2 and state.shape[1]==73
 source=state.double();q=torch.nn.functional.normalize(source[:,3:7],dim=-1)
 w,x,y,z=q.unbind(-1);sine=2*(w*z+x*y);cosine=1-2*(y*y+z*z)
 projection=sine.square()+cosine.square()
 if bool((projection<1e-12).any()):raise ValueError('Yaw frame singular at vertical root heading; no silent fallback')
 theta=torch.atan2(sine,cosine)[:,None];c=torch.cos(theta);s=torch.sin(theta)
 zeros=torch.zeros_like(theta);yaw=torch.cat([torch.cos(theta/2),zeros,zeros,torch.sin(theta/2)],dim=1)
 inverse=yaw*source.new_tensor([1,-1,-1,-1]);local_q=quat_mul(inverse,q)
 local_q=torch.where(local_q[:,:1]<0,-local_q,local_q)
 local=torch.cat([torch.zeros_like(source[:,:2]),source[:,2:3],local_q,rotate_vector(source[:,7:10],c,-s),rotate_vector(source[:,10:13],c,-s),source[:,13:]],dim=1)
 frame=dict(c=c,s=s,yaw=yaw,xy=source[:,:2],quaternion=q,heading_projection_squared=projection)
 return local.to(state.dtype),frame

def restore_state(local,frame,output_dtype=None):
 p=local.double();c=frame['c'];s=frame['s']
 position=rotate_vector(p[:,:3],c,s);position=torch.cat([position[:,:2]+frame['xy'],position[:,2:3]],dim=1)
 q=quat_mul(frame['yaw'],torch.nn.functional.normalize(p[:,3:7],dim=-1))
 q=torch.where((q*frame['quaternion']).sum(-1,keepdim=True)<0,-q,q)
 return torch.cat([position,q,rotate_vector(p[:,7:10],c,s),rotate_vector(p[:,10:13],c,s),p[:,13:]],dim=1).to(local.dtype if output_dtype is None else output_dtype)

class YawCanonicalModel(nn.Module):
 def __init__(self,model,enabled):
  super().__init__();assert enabled in [False,True,0,1];self.model=model;self.enabled=bool(enabled)
 @property
 def state_norm(self):return self.model.state_norm
 def forward(self,state,action,context):
  if not self.enabled:return self.model(state,action,context)
  local,frame=canonicalize_state(state)
  return restore_state(self.model(local.to(self.model.state_norm.mean.dtype),action,context),frame,output_dtype=state.dtype)


def coordinate_metadata(enabled):
 return dict(contract=CONTRACT,enabled=bool(enabled),coordinate_dtype='float64',network_dtype='float32',xy_reference_physical=[0.,0.],quaternion_hemisphere='nonnegative_w',order='raw_model_then_pose_integration',heading_projection_min=1e-12,scope='offline_only')

def apply_coordinate_contract(model,artifact):
 meta=artifact.get('coordinate_contract');plan=artifact.get('calibration_plan',{})
 if meta is None:
  if artifact.get('kind')==KIND or 'yaw_canonical' in plan or 'coordinate_contract' in plan:raise ValueError('Missing declared coordinate contract')
  return model
 if artifact.get('kind')!=KIND:raise ValueError('Coordinate contract requires explicit offline geometry artifact')
 if not isinstance(meta,dict) or type(meta.get('enabled')) is not bool:raise ValueError('Invalid coordinate enabled flag')
 if meta!=coordinate_metadata(meta['enabled']):raise ValueError('Unsupported coordinate contract')
 if plan.get('coordinate_contract')!=meta or type(plan.get('yaw_canonical')) is not bool or plan['yaw_canonical']!=meta['enabled']:raise ValueError('Coordinate plan/metadata mismatch')
 if artifact.get('geometry_contract',{}).get('enabled') is not True:raise ValueError('Coordinate trial requires enabled height geometry')
 return YawCanonicalModel(model,meta['enabled'])
