"""Explicit offline-only geometry input contract; never a policy checkpoint."""
import hashlib
from pathlib import Path
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_sole_clearance import SoleClearance
GEOMETRY_KIND='offline_sole_geometry_dynamics_not_policy_checkpoint'
GEOMETRY_CONTRACT='sole_bottom_height_penetration_mm_v1'
GEOMETRY_VELOCITY_CONTRACT='sole_bottom_height_penetration_velocity_v2'
class GeometryInputLinear(torch.nn.Module):
 def __init__(self,first,norm,geometry,enabled):
  super().__init__();self.geometry=geometry;self.enabled=bool(enabled)
  self.register_buffer('input_mean',norm.mean.detach().clone())
  self.register_buffer('input_std',(norm.var+norm.eps).sqrt().detach().clone())
  self.register_buffer('input_frozen',norm.frozen_mask.detach().clone())
  self.weight=torch.nn.Parameter(torch.cat([first.weight.detach().clone(),first.weight.new_zeros(first.out_features,16)],dim=1))
  self.bias=torch.nn.Parameter(first.bias.detach().clone())
 def geometry_features(self,x):
  raw=x[:,:73]*self.input_std+self.input_mean
  raw=torch.where(self.input_frozen,x[:,:73],raw)
  h=self.geometry(raw)[...,[0,2,4,6]].reshape(len(x),8)/.001
  features=torch.cat([h,torch.relu(-h)],dim=-1)
  return features
 def forward(self,x):
  features=self.geometry_features(x)*float(self.enabled)
  return torch.nn.functional.linear(torch.cat([x,features],dim=-1),self.weight,self.bias)
class GeometryVelocityInputLinear(GeometryInputLinear):
 def __init__(self,first,norm,geometry,velocity_enabled,velocity_scale):
  super().__init__(first,norm,geometry,True)
  self.weight=torch.nn.Parameter(torch.cat([self.weight.detach(),self.weight.new_zeros(first.out_features,8)],dim=1))
  self.velocity_enabled=bool(velocity_enabled);self.velocity_scale=float(velocity_scale)
 def geometry_features(self,x):
  raw=x[:,:73]*self.input_std+self.input_mean
  raw=torch.where(self.input_frozen,x[:,:73],raw)
  raw=raw.clone();raw[:,:2]=0
  height,velocity=self.geometry.height_and_velocity(raw)
  h=height[...,[0,2,4,6]].reshape(len(x),8)/.001
  v=velocity[...,[0,2,4,6]].reshape(len(x),8)/self.velocity_scale
  return torch.cat([h,torch.relu(-h),v*float(self.velocity_enabled)],dim=-1)

def install_geometry_input(model,metadata,cfg):
 assert metadata['contract'] in [GEOMETRY_CONTRACT,GEOMETRY_VELOCITY_CONTRACT] and metadata['height_scale_m']==.001
 assert metadata['ground_height_m']==0 and isinstance(metadata['enabled'],bool)
 asset=Path(metadata['asset'])
 assert hashlib.sha256(asset.read_bytes()).hexdigest()==metadata['asset_sha256']==cfg['extra']['tw_mjcf_sha256']
 if metadata['contract']==GEOMETRY_VELOCITY_CONTRACT:
  from supermimic_sole_corner_velocity import SoleCornerVelocity
  assert metadata['enabled'] is True and isinstance(metadata['velocity_enabled'],bool)
  assert metadata.get('xy_reference_physical',[0.,0.])==[0.,0.]
  assert metadata['policy_dt_s']==.02 and abs(metadata['velocity_scale_m_s']*metadata['policy_dt_s']-.001)<1e-15
  geometry=SoleCornerVelocity(asset,cfg['extra']['tw_state_joint_names']).to(model.net[0].weight.device)
  model.net[0]=GeometryVelocityInputLinear(model.net[0],model.state_norm,geometry,metadata['velocity_enabled'],metadata['velocity_scale_m_s'])
  return model
 geometry=SoleClearance(asset,cfg['extra']['tw_state_joint_names'],0.).to(model.net[0].weight.device)
 model.net[0]=GeometryInputLinear(model.net[0],model.state_norm,geometry,metadata['enabled'])
 return model
def load_geometry_artifact(artifact,device='cuda:0'):
 if artifact.get('kind')!=GEOMETRY_KIND:raise ValueError('Explicit geometry offline artifact kind required')
 cfg=artifact['cfg'];w=artifact['world_model'];meta=artifact['geometry_contract']
 model=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,
  goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),
  termination_hidden=tuple(cfg['wm_term_hidden_dims']))).to(device)
 install_geometry_input(model,meta,cfg);model.load_state_dict(w,strict=True)
 from supermimic_pose_integration_diagnostic import PoseVelocityModel
 return model,PoseVelocityModel(model,artifact['prediction_contract'])
def configure_offline_geometry(model,artifact):
 metadata=artifact.get('geometry_contract')
 if metadata is None:
  if artifact.get('kind')==GEOMETRY_KIND:raise ValueError('Geometry artifact lacks geometry contract')
  return model
 if artifact.get('kind')!=GEOMETRY_KIND:raise ValueError('Unexpected geometry contract on ordinary artifact')
 if 'calibration_plan' in artifact:assert metadata==artifact['calibration_plan']['geometry_contract']
 return install_geometry_input(model,metadata,artifact['cfg'])
