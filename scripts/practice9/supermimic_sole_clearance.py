"""Differentiable box-corner heights; geometry only, never inferred contact forces."""
import hashlib,itertools
from pathlib import Path
import mujoco,numpy as np,torch
from supermimic.algo.kinematics import DifferentiableKinematics,build_kinematic_tree
class SoleClearance(torch.nn.Module):
 def __init__(self,asset,joint_names,ground_height=0.):
  super().__init__();asset=Path(asset);self.asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest()
  m=mujoco.MjModel.from_xml_path(str(asset))
  names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,m.nbody)]
  assert names[0]=='base_link'
  self.fk=DifferentiableKinematics(build_kinematic_tree(m,names,joint_names))
  self.feet=[names.index(side+'_foot') for side in ['left','right']]
  corners=[]
  for side in ['left','right']:
   g=m.geom(side+'_foot_collision')
   assert g.type==mujoco.mjtGeom.mjGEOM_BOX
   np.testing.assert_array_equal(g.quat,[1,0,0,0])
   assert m.geom_bodyid[g.id]==m.body(side+'_foot').id
   corners.append(np.array(list(itertools.product([-1,1],repeat=3)))*g.size+g.pos)
  self.register_buffer('corners',torch.tensor(np.stack(corners),dtype=torch.float32))
  self.ground_height=float(ground_height)
 def forward(self,state):
  assert state.shape[-1]==73
  pos,quat=self.fk(state[:,:3],state[:,3:7],state[:,13:43])
  q=torch.nn.functional.normalize(quat[:,self.feet],dim=-1)
  v=self.corners[None].expand(len(state),-1,-1,-1)
  xyz=q[:,:,None,1:].expand_as(v)
  t=2*torch.cross(xyz,v,dim=-1)
  rotated=v+q[:,:,None,:1]*t+torch.cross(xyz,t,dim=-1)
  return rotated[...,2]+pos[:,self.feet,2,None]-self.ground_height
