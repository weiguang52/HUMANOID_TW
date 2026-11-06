"""Fit-only extra wide root responses; preserve every original response sample."""
import torch
from supermimic_wide_root_response import load_wide_root_response
from supermimic_motion_root_response import derivative
class AdditionalWideRootLoss:
 def __init__(self,root,cfg):
  self.data,self.den,self.axes,self.levels,self.meta=load_wide_root_response(root,cfg,'fit');self.target=derivative(self.data['next_state'],self.den).detach();self.generator=torch.Generator(device='cuda:0').manual_seed(63);self.history=[]
 def __call__(self,model,old):
  assert old.shape==(8,);level=torch.randint(3,(8,),device='cuda:0',generator=self.generator)+2;idx=(old//12)*30+level*6+old%6
  pred=model(*[self.data[k][idx].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  std=(model.state_norm.var[7:13]+model.state_norm.eps).sqrt();std=torch.where(model.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
  err=(derivative(pred,self.den[idx])-self.target[idx])*std[self.axes[idx],None]/std[None,:];value=.5*err.square().mean();assert torch.isfinite(value)
  self.history.append({'update':len(self.history)+1,'original_indices':old.cpu().tolist(),'wide_level_indices':level.cpu().tolist(),'used_indices':idx.cpu().tolist(),'half_weighted_loss':float(value)})
  return value
