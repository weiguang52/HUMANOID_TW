"""Causal loaded-foot history; velocity always comes from current contacts."""
import torch
class ContactHistory:
 def __init__(self,minimum_load=5.,samples=3):
  if samples<1 or not isinstance(samples,int):raise ValueError('positive integer samples required')
  if not 0<minimum_load<float('inf'):raise ValueError('positive finite load required')
  self.minimum_load=minimum_load;self.samples=samples;self.history=None
 def reset(self,ids=None):
  if self.history is not None:
   if ids is None:self.history.zero_()
   else:self.history[:,ids]=False
 def update(self,load,current_point_rms,current_positive_points):
  if load.shape!=current_point_rms.shape or load.shape!=current_positive_points.shape:raise ValueError('shape mismatch')
  if self.history is None:self.history=torch.zeros((self.samples,)+load.shape,device=load.device,dtype=torch.bool)
  if self.history.shape[1:]!=load.shape:raise ValueError('owner shape changed')
  self.history=torch.roll(self.history,1,0);self.history[0]=load>self.minimum_load
  valid=self.history.any(0)&(current_positive_points>0)&(load>1e-6)
  if not torch.isfinite(current_point_rms[valid]).all():raise ValueError('nonfinite current contact velocity')
  return torch.where(valid,current_point_rms,0.)
