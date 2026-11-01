"""All-frame body collision geometry minimum heights/Jacobians, no simulation."""
import itertools
import mujoco,numpy as np,scipy.sparse as sp
class SequenceCollisionHeight:
 def __init__(self,model,qbase,lower_qadr,velocity_columns):
  self.m=model;self.d=mujoco.MjData(model);self.qbase=qbase.copy();self.lower=lower_qadr;self.cols=velocity_columns;self.geoms=[]
  for g in range(model.ngeom):
   if '_collision' not in model.geom(g).name:continue
   if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:verts=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_size[g]
   else:
    assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    k=model.geom_dataid[g];verts=model.mesh_vert[model.mesh_vertadr[k]:model.mesh_vertadr[k]+model.mesh_vertnum[k]].copy()
   self.geoms.append((g,verts))
 def evaluate(self,x,jacobian=True):
  heights=[];blocks=[];x=x.reshape(len(self.qbase),17)
  for i in range(len(x)):
   self.d.qpos[:]=self.qbase[i];self.d.qpos[:3]+=x[i,:3];self.d.qpos[self.lower]+=x[i,3:];mujoco.mj_kinematics(self.m,self.d)
   if jacobian:mujoco.mj_comPos(self.m,self.d)
   local=[]
   for g,verts in self.geoms:
    world=verts@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g];point=world[np.argmin(world[:,2])];heights.append(float(point[2]))
    if jacobian:
     jp=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,jp,None,point,int(self.m.geom_bodyid[g]));local.append(jp[2,self.cols])
   if jacobian:blocks.append(np.array(local))
  return np.array(heights),sp.block_diag(blocks,format='csc') if jacobian else None
