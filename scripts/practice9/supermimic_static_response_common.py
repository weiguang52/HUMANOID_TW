"""Retain static-wide root loss; optionally add root-input/joint-output response."""
import torch
from supermimic_static_wide_root import AdditionalStaticWideRootLoss
class StaticWideResponseCommonLoss(AdditionalStaticWideRootLoss):
 def __init__(self,root,cfg,common_enabled):
  super().__init__(root,cfg);assert common_enabled in [0,1];self.common_enabled=common_enabled
  v=self.data['next_state'];self.joint_target=((v[:,1,43:73]-v[:,0,43:73]+v[:,2,43:73]-v[:,3,43:73])/self.den).detach()
  assert torch.isfinite(self.joint_target).all()
 def __call__(self,model):
  idx=torch.randint(48,(8,),generator=self.generator,device=self.den.device)
  pred=model(*[self.data[k][idx].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  norm=model.state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std)
  response=(pred[:,1,7:13]-pred[:,0,7:13]+pred[:,2,7:13]-pred[:,3,7:13])/self.den[idx]
  root=.5*((response-self.target[idx])*std[7:13][self.axes[idx],None]/std[None,7:13]).square().mean()
  joint_response=(pred[:,1,43:73]-pred[:,0,43:73]+pred[:,2,43:73]-pred[:,3,43:73])/self.den[idx]
  joint=.5*((joint_response-self.joint_target[idx])*std[7:13][self.axes[idx],None]/std[None,43:73]).square().mean()
  error=(pred-self.data['next_state'][idx])/std
  common_linear=.5*error[:,:,7:10].mean(1).square().mean()
  common_angular=.5*error[:,:,10:13].mean(1).square().mean()
  common=common_linear+common_angular
  assert torch.isfinite(root) and torch.isfinite(joint) and torch.isfinite(common)
  self.last_root=root;self.last_joint=joint;self.last_common=common
  self.history.append({'update':len(self.history)+1,'indices':idx.cpu().tolist(),'root_half_loss':float(root),'joint_half_loss':float(joint),'common_linear_half_loss':float(common_linear),'common_angular_half_loss':float(common_angular)})
  return root+joint+self.common_enabled*common
