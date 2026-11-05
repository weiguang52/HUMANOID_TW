"""Dense center objective plus optional root-linear velocity supervision."""
import torch
from supermimic_dense_motion_center import DenseMotionCenterLoss
class LinearDenseCenterLoss(DenseMotionCenterLoss):
 def __init__(self,root,cfg,enabled,device='cuda:0'):
  super().__init__(root,cfg,device);self.enabled=enabled
 def evaluate(self,model,idx):
  d=self.data;pred=model(d['state'][idx],d['action'][idx],d['context'][idx]);std=(model.state_norm.var+model.state_norm.eps).sqrt();std=torch.where(model.state_norm.frozen_mask,torch.ones_like(std),std);error=(pred-d['next_state'][idx])/std
  value=.5*(error[:,10:13].square().mean()+error[:,43:73].square().mean())
  linear=.5*error[:,7:10].square().mean();assert torch.isfinite(value) and torch.isfinite(linear)
  self.last_linear=float(linear.detach());return value+self.enabled*linear
