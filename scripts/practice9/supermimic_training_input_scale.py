"""Training-only first-layer coordinates; original geometry and inference kept."""
import torch
INDICES=list(range(13,43))+list(range(73,103))
RULE='equal_static_motion_fit_rms_min1_joint_action_v1'

def fit_scale(model,static,motion):
 def inputs(data):return torch.cat([model.state_norm.normalize(data['state']),model.action_norm.normalize(data['action']),model.context_norm.normalize(data['context'])],-1)
 x,y=inputs(static),inputs(motion)
 assert x.shape==(3305,137) and y.shape==(828,137)
 idx=torch.tensor(INDICES,device=x.device)
 # FP64 reduction avoids fit-statistic dependence on chunk/reduction roundoff.
 rms=(.5*x[:,idx].double().square().mean(0)+.5*y[:,idx].double().square().mean(0)).sqrt().clamp_min(1).float()
 assert torch.isfinite(rms).all()
 scale=torch.ones(153,device=x.device);scale[idx]=rms
 return scale

class TrainingInputScale:
 def __init__(self,first,scale,enabled):
  assert first.weight.shape[1]==153 and scale.shape==(153,)
  assert not hasattr(first,'_training_input_scale_active')
  self.first=first;self.scale=scale.detach().clone();self.enabled=bool(enabled)
  self.original_forward=first.forward
  if self.enabled:
   with torch.no_grad():first.weight.mul_(self.scale)
   def forward(x):
    features=first.geometry_features(x)*float(first.enabled)
    full=torch.cat([x,features],-1)
    return torch.nn.functional.linear(full/self.scale,first.weight,first.bias)
   first.forward=forward;first._training_input_scale_active=True
 def export(self):
  if self.enabled:
   with torch.no_grad():self.first.weight.div_(self.scale)
   self.first.forward=self.original_forward
   del self.first._training_input_scale_active
   self.enabled=False
