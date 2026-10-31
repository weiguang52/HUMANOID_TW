"""Explicit offline prediction structures; never implicitly selected by a policy."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_mul
from supermimic.algo.networks import _rotvec_to_quat
POSE_KIND='offline_pose_velocity_calibration_not_policy_checkpoint'
CONTRACT='pose_from_velocity_fit_alpha_v1'
class PoseVelocityModel(nn.Module):
 def __init__(self,model,metadata):
  super().__init__();assert metadata['contract']==CONTRACT
  digest=metadata['fit_audit_sha256'];assert isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
  self.model=model;self.metadata=dict(metadata);self.alpha=dict(metadata['alpha']);self.dt=float(metadata['dt'])
  assert set(self.alpha)=={'translation','rotation','joint'} and self.dt==.02 and all(isinstance(v,(float,int)) and 0<=v<=1 for v in self.alpha.values())
 def forward(self,state,action,context):
  pred=self.model(state,action,context);a=self.alpha;dt=self.dt
  position=state[:,:3]+dt*((1-a['translation'])*state[:,7:10]+a['translation']*pred[:,7:10])
  omega=(1-a['rotation'])*state[:,10:13]+a['rotation']*pred[:,10:13]
  quat=quat_mul(_rotvec_to_quat(dt*omega),state[:,3:7]);quat=torch.nn.functional.normalize(quat,dim=-1)
  joints=state[:,13:43]+dt*((1-a['joint'])*state[:,43:73]+a['joint']*pred[:,43:73])
  return torch.cat([position,quat,pred[:,7:13],joints,pred[:,43:73]],dim=-1)

class PoseIntegrationDiagnostic(PoseVelocityModel):
 def __init__(self,model,audit,checkpoint_sha256):
  path=Path(audit);d=json.loads(path.read_text());assert d['candidate_sha256']==checkpoint_sha256
  metadata=dict(contract=CONTRACT,diagnostic_only=True,weights_sha256=checkpoint_sha256,fit_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),alpha=d['alpha'],dt=float(d['plan']['dt']),scope='Fit-only coefficients, frozen weights, predicted endpoint velocities only; not policy promotion')
  super().__init__(model,metadata)

def apply_offline_prediction_contract(model,artifact):
 metadata=artifact.get('prediction_contract')
 if artifact.get('kind') in [POSE_KIND,'offline_sole_geometry_dynamics_not_policy_checkpoint']:
  if not isinstance(metadata,dict) or metadata.get('contract')!=CONTRACT:raise ValueError('Pose-velocity artifact requires its explicit prediction contract')
  return PoseVelocityModel(model,metadata)
 if metadata is not None:raise ValueError('Unexpected prediction contract on an ordinary model')
 return model
