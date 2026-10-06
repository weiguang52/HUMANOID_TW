"""TW195 offline contact projection. No contact relabeling or time rescaling."""
import json,pathlib,xml.etree.ElementTree as ET
import numpy as np,pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from contact_labels import from_features,resample
URDF='/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf'
HUMAN=pathlib.Path('/root/gpufree-data/datasets/practice9/humanml3d_rebuild/staging-v1/HumanML3D')

def phases(mid,n):
 f=np.load(HUMAN/'new_joint_vecs'/f'{mid}.npy');j=np.load(HUMAN/'new_joints'/f'{mid}.npy')
 c,k,_,raw=from_features(f,j);c,k=resample(c,k,n)
 t=np.linspace(0,len(raw)-1,n);lo=np.floor(t).astype(int);hi=np.ceil(t).astype(int)
 full=np.stack([raw[:,:2].all(1),raw[:,2:].all(1)],1)
 return c.astype(bool),k.astype(bool),(full[lo]&full[hi])

def ramps(mask,width=5):
 out=np.zeros(mask.shape,float)
 for j in range(2):
  for i in range(len(mask)):
   if mask[i,j]:out[i,j]=min(1,(out[i-1,j]+1/width) if i else 1/width)
  for i in range(len(mask)-2,-1,-1):
   if mask[i,j]:out[i,j]=min(out[i,j],out[i+1,j]+1/width)
 return out

class Model:
 def __init__(self,names,bodies):
  self.m=pin.buildModelFromUrdf(URDF,pin.JointModelFreeFlyer());self.d=self.m.createData()
  self.names=list(names);self.bodies=list(bodies)
  self.qidx=np.array([self.m.joints[self.m.getJointId(str(n))].idx_q for n in names])
  self.vidx=np.array([self.m.joints[self.m.getJointId(str(n))].idx_v for n in names])
  self.bids=[self.m.getFrameId(str(n),pin.BODY) for n in bodies]
  self.feet=[self.m.getFrameId(n,pin.BODY) for n in ['left_foot','right_foot']]
  self.sel=[i for i,n in enumerate(names) if str(n) in [s+'_'+k for s in ['left','right'] for k in ['hip_pitch_joint','hip_roll_joint','knee_pitch_joint','ankle_pitch_joint','foot_roll']]]
  self.cols=self.vidx[self.sel]
  r=ET.parse(URDF).getroot();self.corners=[];self.com=[]
  for n in ['left_foot','right_foot']:
   b=r.find(f"link[@name='{n}']/collision");size=np.fromstring(b.find('geometry/box').get('size'),sep=' ');o=np.fromstring(b.find('origin').get('xyz'),sep=' ')
   self.corners.append(np.array([o+size*np.array([x,y,-1])/2 for x,y in [(-1,-1),(-1,1),(1,-1),(1,1)]]))
  for n in bodies:
   o=r.find(f"link[@name='{n}']/inertial/origin");self.com.append(np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' '))
 def configuration(self,a,t):
  q=pin.neutral(self.m);q[:3]=a['root_pos'][t];q[3:7]=a['root_quat_wxyz'][t][[1,2,3,0]];q[self.qidx]=a['joint_pos'][t];return q
 def fk(self,q,jac=False):
  if jac:pin.computeJointJacobians(self.m,self.d,q)
  else:pin.forwardKinematics(self.m,self.d,q)
  pin.updateFramePlacements(self.m,self.d)
 def foot(self,j,jac=False):
  f=self.d.oMf[self.feet[j]];p=self.corners[j]@f.rotation.T+f.translation
  if not jac:return p
  J=pin.getFrameJacobian(self.m,self.d,self.feet[j],pin.LOCAL_WORLD_ALIGNED)
  jacob=[]
  for c in self.corners[j]:
   jp=J[:3]-pin.skew(f.rotation@c)@J[3:];jacob.append(np.column_stack(([0,0,1],jp[:,self.cols])))
  return p,np.array(jacob)

def project(a,mid):
 model=Model(a['joint_names'],a['body_names']);n=len(a['joint_pos']);c,k,full=phases(mid,n);active=c&k;weight=ramps(active);flatweight=ramps(active&full)
 original=[]
 for t in range(n):model.fk(model.configuration(a,t));original.append([model.foot(j) for j in range(2)])
 original=np.array(original);anchors=original.mean(2)[:,:,:2].copy()
 for j in range(2):
  mask=active[:,j]&full[:,j];edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
  for start,end in zip(starts,ends):anchors[start:end,j]=np.median(anchors[start:end,j],axis=0)
 qout=[];root=[];prevdelta=np.zeros(11);prevq=None
 lower=model.m.lowerPositionLimit[model.qidx[model.sel]];upper=model.m.upperPositionLimit[model.qidx[model.sel]];vel=model.m.velocityLimit[model.cols];dt=1/float(a['fps'][0]);evaluations=[]
 for t in range(n):
  base=model.configuration(a,t);orig=base[model.qidx[model.sel]]
  lb=np.r_[-.03,lower-orig];ub=np.r_[.03,upper-orig]
  if prevq is not None:
   lb[1:]=np.maximum(lb[1:],prevq-vel*dt-orig);ub[1:]=np.minimum(ub[1:],prevq+vel*dt-orig)
  cache={}
  def calc(x):
   if 'x' in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
   q=base.copy();q[2]+=x[0];q[model.qidx[model.sel]]+=x[1:];model.fk(q,True)
   scale=np.r_[.02,np.repeat(.3,10)];smooth=np.r_[.006,np.repeat(.12,10)]
   rs=[x/scale,(x-prevdelta)/smooth];js=[np.diag(1/scale),np.diag(1/smooth)]
   for j in range(2):
    p,J=model.foot(j,True);imin=np.argmin(p[:,2]);w=weight[t,j];fw=flatweight[t,j]
    rs.append(np.array([(p[imin,2]-.001)*w/.0008]));js.append(J[imin,2:3]*w/.0008)
    rs.append((p[:,2]-p[:,2].mean())*fw/.001);js.append((J[:,2]-J[:,2].mean(0))*fw/.001)
    rs.append((p.mean(0)[:2]-anchors[t,j])*fw/.008);js.append(J.mean(0)[:2]*fw/.008)
    penetration=min(p[imin,2]-.0002,0);rs.append(np.array([penetration/.0003]));js.append(J[imin,2:3]/.0003 if penetration<0 else np.zeros((1,11)))
   cache.update(x=x.copy(),r=np.concatenate(rs),J=np.vstack(js));return cache['r'],cache['J']
  x0=np.clip(prevdelta,lb+1e-10,ub-1e-10)
  result=least_squares(lambda x:calc(x)[0],x0,jac=lambda x:calc(x)[1],bounds=(lb,ub),max_nfev=15,ftol=1e-5,xtol=1e-5,gtol=1e-5)
  x=result.x;out=a['joint_pos'][t].copy();out[model.sel]+=x[1:];qout.append(out);rp=a['root_pos'][t].copy();rp[2]+=x[0];root.append(rp);prevdelta=x;prevq=out[model.sel];evaluations.append(result.nfev)
 return np.array(qout),np.array(root),dict(mean_nfev=float(np.mean(evaluations)),contact=c,known=k,full=full,original=original)

def rebuild(a,joints,root):
 out={key:val.copy() for key,val in a.items()};out['joint_pos']=joints;out['root_pos']=root;model=Model(a['joint_names'],a['body_names']);n=len(joints);dt=1/float(a['fps'][0]);positions=[];quats=[];centers=[];soles=[]
 for t in range(n):
  model.fk(model.configuration(out,t));frames=[model.d.oMf[i] for i in model.bids]
  positions.append([f.translation.copy() for f in frames]);quats.append([Rotation.from_matrix(f.rotation).as_quat()[[3,0,1,2]] for f in frames]);centers.append([f.translation+f.rotation@com for f,com in zip(frames,model.com)]);soles.append([model.foot(j) for j in range(2)])
 p=np.array(positions);quat=np.array(quats);centers=np.array(centers);omega=np.zeros_like(p)
 for t in range(1,n):
  flip=np.sum(quat[t]*quat[t-1],axis=-1)<0;quat[t,flip]*=-1
 for j in range(len(model.bids)):
  rot=Rotation.from_quat(quat[:,j][:,[1,2,3,0]]);w=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt);omega[:,j]=np.vstack((w[:1],w,w[-1:]))
 out.update(joint_vel=np.gradient(joints,dt,axis=0).astype(np.float32),body_pos_w=p.astype(np.float32),body_quat_w=quat.astype(np.float32),body_lin_vel_w=np.gradient(centers,dt,axis=0).astype(np.float32),body_ang_vel_w=omega.astype(np.float32))
 return out,np.array(soles)

def process(row,output):
 import time
 start=time.time();a=dict(np.load(row['file']));q,root,meta=project(a,row['id']);out,points=rebuild(a,q,root);old=meta.pop('original');c=meta.pop('contact');k=meta.pop('known');full=meta.pop('full');stance=c&k;flat=stance&full;z=points[:,:,:,2].min(2);oldz=old[:,:,:,2].min(2);model=Model(a['joint_names'],a['body_names'])
 def quant(x):return float(np.quantile(x,.95)) if x.size else None
 audit=dict(id=row['id'],frames=len(q),seconds=time.time()-start,**meta,old_stance_z_p95=quant(oldz[stance]),new_stance_z_p95=quant(z[stance]),stance_error_p95=quant(abs(z[stance]-.001)),flat_spread_p95=quant(np.ptp(points[:,:,:,2],axis=2)[flat]),min_sole=float(z.min()),max_q_change=float(abs(q-a['joint_pos']).max()),max_root_z_change=float(abs(root[:,2]-a['root_pos'][:,2]).max()),velocity_ratio_max=float((abs(out['joint_vel'])/model.m.velocityLimit[model.vidx]).max()),upper_joint_change=float(abs(q[:,14:]-a['joint_pos'][:,14:]).max()),contact_unchanged=True,timing_unchanged=True)
 output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**out);output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2));return audit

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--ids',required=True);p.add_argument('--output',required=True);args=p.parse_args();m=json.load(open(args.manifest));ids=args.ids.split(',')
 for r in m['motions']:
  if r['id'] in ids:print(json.dumps(process(r,pathlib.Path(args.output)/(r['id']+'.npz'))),flush=True)
