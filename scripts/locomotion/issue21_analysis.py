"""Independent NumPy residuals and threshold crossing extraction."""
import numpy as np,json
from pathlib import Path
from issue21_common import REPO
GROUPS={'ang_vel':(0,3),'gravity':(3,6),'commands':(6,9),'joint_pos_obs':(9,39),'joint_vel_obs':(39,69),'last_action':(69,99),'gait_phase':(99,101)}
class Mirror:
 def __init__(self,joints,bodies):
  fixture=json.loads((REPO/'tests/locomotion/fixtures/issue16_joint_axes.json').read_text());rows={x['joint']:x for x in fixture['joints']};self.jp=np.array([joints.index(rows[n]['partner']) for n in joints]);self.js=np.array([rows[n]['sign'] for n in joints]);self.bp=np.array([bodies.index(('right_'+n[5:]) if n.startswith('left_') else ('left_'+n[6:]) if n.startswith('right_') else n) for n in bodies])
 def joint(self,x):return x[...,self.jp]*self.js
 def obs(self,x):
  y=x.copy();y[...,:3]*=[-1,1,-1];y[...,3:6]*=[1,-1,1];y[...,6:9]*=[1,-1,-1]
  for start in [9,39,69]:y[...,start:start+30]=self.joint(x[...,start:start+30])
  y[...,99:101]*=-1;return y
 def root(self,x):
  y=x.copy();y[...,:3]*=[1,-1,1];y[...,3:7]*=[1,-1,1,-1];y[...,7:10]*=[1,-1,1];y[...,10:13]*=[-1,1,-1];return y
 def contact(self,x):return x[...,self.bp,:]*np.array([1,-1,1])
 def residuals(self,a,b):
  result={};o=b['obs'].astype(np.float64)-self.obs(a['obs'].astype(np.float64))
  for name,(l,h) in GROUPS.items():result[name]=o[...,l:h]
  result['action']=b['action'].astype(np.float64)-self.joint(a['action'].astype(np.float64))
  expected=self.root(a['root'].astype(np.float64));actual=b['root'].astype(np.float64).copy();q=actual[...,3:7];qm=expected[...,3:7];actual[...,3:7]=np.where((q*qm).sum(-1,keepdims=True)<0,-q,q)
  for n,l,h in [('root_position',0,3),('root_quaternion',3,7),('root_linear_velocity',7,10),('root_angular_velocity',10,13)]:result[n]=actual[...,l:h]-expected[...,l:h]
  for name in ['joint_pos','joint_vel','applied_torque','estimated_pd_torque','explicit_actuation_force','joint_target']:result[name]=b[name].astype(np.float64)-self.joint(a[name].astype(np.float64))
  result['contact']=b['contact'].astype(np.float64)-self.contact(a['contact'].astype(np.float64));return result

def crossings(x,names=None,offset=0):
 x=np.asarray(x,dtype=np.float64);flat=x.reshape(x.shape[0],x.shape[1],-1);out=[]
 for threshold in [1e-6,1e-4,1e-2]:
  hits=np.abs(flat)>threshold;steps=np.flatnonzero(hits.any(axis=(1,2)))
  def entry(env=None):
   h=hits if env is None else hits[:,env:env+1];ts=np.flatnonzero(h.any(axis=(1,2)))
   if not len(ts):return None
   t=int(ts[0]);e,c=np.unravel_index(np.abs(flat[t] if env is None else flat[t,env:env+1]).argmax(),flat[t].shape if env is None else flat[t,env:env+1].shape);e=e if env is None else env
   return {'step':t+offset,'env':int(e),'component':names[c] if names else int(c),'signed_residual':float(flat[t,e,c]),'absolute_residual':float(abs(flat[t,e,c]))}
  out.append({'threshold':threshold,'first':entry(),'per_env':[entry(e) for e in range(flat.shape[1])]})
 return out

def slope(yaw,terminated,dt=.02):
 y=np.unwrap(np.asarray(yaw,dtype=np.float64),axis=0);t=np.arange(1,len(y)+1)*dt;keep=(t>=5)&(t<=20);x=t[keep];x=x-x.mean();v=(x[:,None]*y[keep]).sum(0)/(x*x).sum()*180/np.pi;valid=~np.asarray(terminated,dtype=bool).any(0);v[~valid]=np.nan;return v,valid

def growth(error,first_contact,dt=.02):
 x=np.asarray(error,dtype=float);out=[]
 for e,c in enumerate(first_contact):
  if c is None:out.append(None);continue
  start=int(c);stop=min(len(x),start+round(2/dt));hits=np.flatnonzero(x[start:stop,e]>=.01)
  if len(hits):stop=start+int(hits[0])+1
  v=x[start:stop,e];idx=np.flatnonzero((v>0)&np.isfinite(v));fit=None
  if len(idx)>=10:
   t=idx*dt;z=np.log(v[idx]);coef=np.polyfit(t,z,1);pred=np.polyval(coef,t);ss=float(((z-z.mean())**2).sum());r2=1-float(((z-pred)**2).sum())/ss if ss else 0;fit={'rate_per_second':float(coef[0]),'r2':r2,'points':len(idx),'exponential':bool(coef[0]>0 and r2>=.8)}
  out.append(fit)
 return out
