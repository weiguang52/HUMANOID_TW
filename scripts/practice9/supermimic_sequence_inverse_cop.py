"""Prescribed-q inverse COP and five-colour local temporal Jacobian. No integration."""
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
class SequenceInverseCOP:
 def __init__(self,asset,qbase,lower_qadr,joint_qadr,joint_vadr,dt):
  self.m=mujoco.MjModel.from_xml_path(str(asset));self.m.geom_contype[:]=0;self.m.geom_conaffinity[:]=0;self.d=mujoco.MjData(self.m)
  self.qbase=qbase.copy();self.qa=joint_qadr;self.va=joint_vadr;self.lower=lower_qadr;self.dt=dt;self.n=len(qbase);self.rot=Rotation.from_quat(qbase[:,[4,5,6,3]])
  w=np.zeros((self.n,3));w[1:-1]=(self.rot[2:]*self.rot[:-2].inv()).as_rotvec()/(2*dt);w[0]=w[1];w[-1]=w[-2];self.local_w=self.rot.inv().apply(w)
 def evaluate(self,x):
  q=self.qbase.copy();q[:,:3]+=x[:,:3];q[:,self.lower]+=x[:,3:];v=np.zeros((self.n,self.m.nv));v[:,:3]=np.gradient(q[:,:3],self.dt,axis=0);v[:,3:6]=self.local_w;v[:,self.va]=np.gradient(q[:,self.qa],self.dt,axis=0);acc=np.gradient(v,self.dt,axis=0);cop=[];forces=[]
  for i in range(self.n):
   self.d.qpos[:]=q[i];self.d.qvel[:]=v[i];self.d.qacc[:]=acc[i];mujoco.mj_inverse(self.m,self.d);f=self.d.qfrc_inverse[:3].copy();t=self.rot[i].apply(self.d.qfrc_inverse[3:6])+np.cross(q[i,:3],f);assert abs(f[2])>1e-8;cop.append(np.array([-t[1],t[0]])/f[2]);forces.append(f)
  return np.array(cop),np.array(forces)
 def jacobian(self,x,epsilon=1e-6):
  # q->central qvel->qacc has radius2; same-colour input columns are five frames apart.
  # One-sided np.gradient endpoints also have radius2; hence no overlapping supports.
  rr=[];cc=[];vv=[]
  for colour in range(5):
   frames=np.arange(colour,self.n,5)
   for j in range(17):
    dx=np.zeros_like(x);dx[frames,j]=epsilon;plus=self.evaluate(x+dx)[0];minus=self.evaluate(x-dx)[0];derivative=(plus-minus)/(2*epsilon)
    for k in frames:
     for output in range(max(0,k-2),min(self.n,k+3)):
      for axis in range(2):rr.append(2*output+axis);cc.append(17*k+j);vv.append(derivative[output,axis])
  return coo_matrix((vv,(rr,cc)),shape=(2*self.n,17*self.n)).tocsc()
