"""Centroidal wrench inferred from full-body inertial-frame FK trajectories."""
import numpy as np
from scipy.spatial.transform import Rotation

def infer_wrench(pos,rot,mass,inertia_principal,dt,gravity):
 total=mass.sum();n,bodies=pos.shape[:2]
 com=np.einsum('tbj,b->tj',pos,mass)/total
 vel=np.gradient(pos,dt,axis=0);vcom=np.gradient(com,dt,axis=0)
 acc=np.gradient(vcom,dt,axis=0);omega=np.zeros((n,bodies,3))
 for b in range(bodies):
  rr=Rotation.from_matrix(rot[:,b]);omega[1:-1,b]=(rr[2:]*rr[:-2].inv()).as_rotvec()/(2*dt)
  omega[0,b]=omega[1,b];omega[-1,b]=omega[-2,b]
 inertia=np.einsum('tbij,bj,tbkj->tbik',rot,inertia_principal,rot)
 spin=np.einsum('tbij,tbj->tbi',inertia,omega)
 orbital=np.cross(pos-com[:,None],mass[None,:,None]*(vel-vcom[:,None]))
 momentum=(spin+orbital).sum(1);hdot=np.gradient(momentum,dt,axis=0)
 force=total*(acc-gravity);torque=hdot+np.cross(com,force)
 return com,momentum,force,torque

def ground_cop(force,torque):
 assert np.all(np.abs(force[:,2])>1e-12),'Ground COP undefined for zero vertical force'
 return np.column_stack([-torque[:,1],torque[:,0]])/force[:,2,None]
