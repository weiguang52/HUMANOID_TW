"""Fixed-alpha pose reconstruction using true velocities; not deployed predictions."""
import argparse,hashlib,json,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;origin=r/'antithetic_integration_consistency_v1/audit.json';fixed=json.loads(origin.read_text());alpha=fixed['alpha'];assert set(alpha)=={'translation','rotation','joint'} and all(0<=x<=1 for x in alpha.values());dt=.02

def normalize(q):return q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-15)
def mul(x,y):return np.concatenate([x[:,:1]*y[:,:1]-np.sum(x[:,1:]*y[:,1:],axis=1,keepdims=True),x[:,:1]*y[:,1:]+y[:,:1]*x[:,1:]+np.cross(x[:,1:],y[:,1:])],axis=1)
def expquat(v):
 t=np.linalg.norm(v,axis=1,keepdims=True)
 return np.concatenate([np.cos(t/2),.5*np.sinc(t/(2*np.pi))*v],axis=1)
def step(state,true_next):
 out=true_next.copy()
 out[:,:3]=state[:,:3]+dt*((1-alpha['translation'])*state[:,7:10]+alpha['translation']*true_next[:,7:10])
 out[:,13:43]=state[:,13:43]+dt*((1-alpha['joint'])*state[:,43:73]+alpha['joint']*true_next[:,43:73])
 omega=(1-alpha['rotation'])*state[:,10:13]+alpha['rotation']*true_next[:,10:13]
 out[:,3:7]=normalize(mul(expquat(dt*omega),normalize(state[:,3:7])))
 np.testing.assert_array_equal(out[:,7:13],true_next[:,7:13]);np.testing.assert_array_equal(out[:,43:73],true_next[:,43:73])
 return out

def metrics(pred,target):
 q=normalize(pred[:,3:7]);t=normalize(target[:,3:7]);t=np.where(np.sum(q*t,axis=1,keepdims=True)<0,-t,t)
 angle=4*np.arctan2(np.linalg.norm(q-t,axis=1),np.linalg.norm(q+t,axis=1))
 return dict(root_position_error_p95_m=float(np.quantile(np.linalg.norm(pred[:,:3]-target[:,:3],axis=1),.95)),root_orientation_error_p95_rad=float(np.quantile(angle,.95)),joint_position_rmse_rad=float(np.sqrt(np.mean((pred[:,13:43]-target[:,13:43])**2))))
# Independent constant-velocity identity case in each axis.
s=np.zeros((3,73));s[:,3]=1;s[:,7:10]=np.eye(3);s[:,10:13]=np.eye(3)*.2;s[:,43:73]=.1
expected=s.copy();expected[:,:3]=dt*np.eye(3);expected[:,13:43]=dt*.1;expected[:,3:7]=expquat(dt*s[:,10:13]);np.testing.assert_allclose(step(s,s),expected,rtol=0,atol=1e-12)
rows=[];sources=[]
for group,relative,count in [('static','static_perturb_v1',16),('motion','motion_pd_collection_cache_v2/full',32),('branch','orthogonal_branches_v1',40)]:
 collection=r/relative;items=json.loads((collection/'results.json').read_text())['rows'];assert len(items)==count
 for item in items:
  name=item['file'] if group=='branch' else item['trace'];path=collection/name;assert path.resolve().is_relative_to(collection.resolve());sha=hashlib.sha256(path.read_bytes()).hexdigest()
  if 'sha256' in item:assert sha==item['sha256']
  with np.load(path,allow_pickle=False) as z:state=z['state'].astype(np.float64);nxt=z['next_state'].astype(np.float64)
  n=len(state);assert state.shape==nxt.shape==(n,73) and np.isfinite(state).all() and np.isfinite(nxt).all()
  if group!='branch':np.testing.assert_array_equal(nxt[:-1],state[1:])
  sources.append(dict(group=group,file=name,sha256=sha))
  values={}
  for h in ([1] if group=='branch' else [1,8,32]):
   windows=n-h+1
   if windows<=0:values[str(h)]=dict(windows=0);continue
   pred=state[:windows].copy()
   for offset in range(h):pred=step(pred,nxt[offset:offset+windows])
   assert np.isfinite(pred).all();values[str(h)]=dict(windows=windows,**metrics(pred,nxt[h-1:h-1+windows]))
  rows.append(dict(group=group,split=item['split'],file=name,frames=n,oracle=values))
summary=[]
for group in ['static','motion','branch']:
 for split in ['fit','validation']:
  for h in ([1] if group=='branch' else [1,8,32]):
   selected=[x['oracle'][str(h)] for x in rows if x['group']==group and x['split']==split and x['oracle'][str(h)]['windows']>0]
   summary.append(dict(group=group,split=split,horizon=h,eligible_episodes=len(selected),windows=sum(x['windows'] for x in selected),medians={k:statistics.median(x[k] for x in selected) for k in ['root_position_error_p95_m','root_orientation_error_p95_rad','joint_position_rmse_rad']}))
out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),alpha=alpha,alpha_source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),constant_velocity_identity_passed=True,sources=sources,rows=rows,summary=summary,scope='Nondeployable true-endpoint-velocity oracle with original fixed alpha, no refit. Not a global error lower bound: biased predicted velocities can compensate pose error. No reset crossings/padding/filtering, no model/policy fitting or gate changes.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(summary),flush=True)
