"""Differentiable sole corner vertical velocities, without contact-force inference."""
import torch
from supermimic_sole_clearance import SoleClearance
class SoleCornerVelocity(torch.nn.Module):
 def __init__(self,asset,joint_names):
  super().__init__();self.geometry=SoleClearance(asset,joint_names,0.)
 def forward(self,state):
  geometry=self.geometry
  pos,quat,linear,angular=geometry.fk.forward_with_velocity(state[:,:3],state[:,3:7],state[:,7:10],state[:,10:13],state[:,13:43],state[:,43:73])
  q=torch.nn.functional.normalize(quat[:,geometry.feet],dim=-1)
  v=geometry.corners[None].expand(len(state),-1,-1,-1)
  xyz=q[:,:,None,1:].expand_as(v);t=2*torch.cross(xyz,v,dim=-1)
  rotated=v+q[:,:,None,:1]*t+torch.cross(xyz,t,dim=-1)
  velocity=linear[:,geometry.feet,None,:]+torch.cross(angular[:,geometry.feet,None,:].expand_as(rotated),rotated,dim=-1)
  return velocity[...,2]
