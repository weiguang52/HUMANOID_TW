"""Explicit offline diagnostic wrapper; never selected by training or inference."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from mjlab.utils.lab_api.math import quat_mul
from supermimic.algo.networks import _rotvec_to_quat
class PoseIntegrationDiagnostic(nn.Module):
 def __init__(self,model,audit,checkpoint_sha256):
  super().__init__();path=Path(audit);d=json.loads(path.read_text());assert d['candidate_sha256']==checkpoint_sha256
  self.model=model;self.alpha=d['alpha'];self.dt=float(d['plan']['dt']);assert self.dt==.02 and all(0<=v<=1 for v in self.alpha.values())
  self.metadata=dict(diagnostic_only=True,weights_sha256=checkpoint_sha256,fit_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),alpha=self.alpha,dt=self.dt,scope='Fit-only coefficients, frozen weights, predicted endpoint velocities only; not production or policy promotion')
 def forward(self,state,action,context):
  pred=self.model(state,action,context);a=self.alpha;dt=self.dt
  position=state[:,:3]+dt*((1-a['translation'])*state[:,7:10]+a['translation']*pred[:,7:10])
  omega=(1-a['rotation'])*state[:,10:13]+a['rotation']*pred[:,10:13]
  quat=quat_mul(_rotvec_to_quat(dt*omega),state[:,3:7]);quat=torch.nn.functional.normalize(quat,dim=-1)
  joints=state[:,13:43]+dt*((1-a['joint'])*state[:,43:73]+a['joint']*pred[:,43:73])
  return torch.cat([position,quat,pred[:,7:13],joints,pred[:,43:73]],dim=-1)
