"""Read-only pose integration feasibility; real terminal velocities are oracle data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__)
for key in ['trajectories','branches','candidate','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();dt=.02
plan=dict(dt=dt,fixed_alpha=[.5,1.],fit_alpha='One scalar per translation/world-rotation/joint group, all fit sources only, clipped[0,1]',scope='Read-only; actual terminal-velocity reconstruction is an oracle feasibility diagnostic, not deployed prediction or control acceptance')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
def normalize(q):return q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-15)
def mul(a,b):return np.concatenate([a[:,:1]*b[:,:1]-np.sum(a[:,1:]*b[:,1:],axis=-1,keepdims=True),a[:,:1]*b[:,1:]+b[:,:1]*a[:,1:]+np.cross(a[:,1:],b[:,1:])],axis=-1)
def relative_rotvec(s,n):
 q=normalize(s[:,3:7]);qn=normalize(n[:,3:7]);qc=q.copy();qc[:,1:]*=-1;delta=normalize(mul(qn,qc));delta=np.where(delta[:,:1]<0,-delta,delta);v=delta[:,1:];length=np.linalg.norm(v,axis=-1,keepdims=True);return v*(2*np.arctan2(length,delta[:,:1])/np.maximum(length,1e-15))
def expquat(v):
 angle=np.linalg.norm(v,axis=-1,keepdims=True);return np.concatenate([np.cos(angle/2),.5*np.sinc(angle/(2*np.pi))*v],axis=-1)
records=[]
for source,root in [('trajectory',a.trajectories),('branch',a.branches)]:
 report=json.loads((root/'results.json').read_text())
 if source=='branch':assert report['contract']=='branch_transitions_not_episodes_v1'
 for row in report['rows']:
  file=row['trace'] if source=='trajectory' else row['file'];path=(root/file).resolve();assert path.is_relative_to(root.resolve())
  with np.load(path,allow_pickle=False) as z:d={k:z[k].astype(np.float64) for k in ['state','next_state','action','context']}
  for k,width in [('state',73),('next_state',73),('action',30),('context',34)]:assert d[k].shape==(len(d['state']),width) and np.isfinite(d[k]).all()
  records.append(dict(source=source,split=row['split'],file=file,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data=d))
fit=[r for r in records if r['split']=='fit'];s=np.concatenate([r['data']['state'] for r in fit]);n=np.concatenate([r['data']['next_state'] for r in fit])
rate={'translation':(n[:,:3]-s[:,:3])/dt,'rotation':relative_rotvec(s,n)/dt,'joint':(n[:,13:43]-s[:,13:43])/dt};vel={'translation':slice(7,10),'rotation':slice(10,13),'joint':slice(43,73)};alphas={};raw={}
for group,sl in vel.items():
 dv=n[:,sl]-s[:,sl];den=np.sum(dv**2);assert den>0;value=float(np.sum(dv*(rate[group]-s[:,sl]))/den);raw[group]=value;alphas[group]=float(np.clip(value,0,1))
# Convention checks independent of any empirical error outcome.
q=np.tile(np.array([1.,0,0,0]),(3,1));rv=np.eye(3)*.2;test=np.zeros((3,73));test[:,3:7]=q;next_test=test.copy();next_test[:,3:7]=mul(expquat(rv),q);np.testing.assert_allclose(relative_rotvec(test,next_test),rv,rtol=0,atol=1e-12)
model_data=torch.load(a.candidate,map_location='cpu',weights_only=False);assert model_data['kind']=='offline_dynamics_calibration_not_policy_checkpoint';w=model_data['world_model'];cfg=model_data['cfg'];model=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();model.load_state_dict(w);model.eval()
for root in [a.trajectories,a.branches]:
 m=json.loads((root/'control_mapping.json').read_text());assert m['state_joint_names']==cfg['extra']['tw_state_joint_names'] and m['action_joint_names']==cfg['extra']['tw_action_joint_names']
def integrate(s,end,alpha):
 out=end.copy();out[:,:3]=s[:,:3]+dt*((1-alpha['translation'])*s[:,7:10]+alpha['translation']*end[:,7:10]);out[:,13:43]=s[:,13:43]+dt*((1-alpha['joint'])*s[:,43:73]+alpha['joint']*end[:,43:73]);omega=(1-alpha['rotation'])*s[:,10:13]+alpha['rotation']*end[:,10:13];out[:,3:7]=normalize(mul(expquat(dt*omega),normalize(s[:,3:7])));return out
rules={'trapezoid':dict.fromkeys(vel,.5),'terminal':dict.fromkeys(vel,1.),'fit_only':alphas};results=[]
for source in ['trajectory','branch']:
 for split in ['fit','validation']:
  group=[r for r in records if r['source']==source and r['split']==split];data={k:np.concatenate([r['data'][k] for r in group]) for k in ['state','next_state','action','context']};s,n=data['state'],data['next_state'];pred=[]
  for start in range(0,len(s),1024):
   b={k:torch.tensor(v[start:start+1024],device='cuda:0',dtype=torch.float32) for k,v in data.items()}
   with torch.no_grad():value=model(b['state'],b['action'],b['context']).cpu().numpy().astype(np.float64)
   assert np.isfinite(value).all();pred.append(value)
  pred=np.concatenate(pred);variants={'model_independent_pose':pred}
  for name,alpha in rules.items():variants['oracle_'+name]=integrate(s,n,alpha);variants['model_velocity_'+name]=integrate(s,pred,alpha)
  out={}
  for name,value in variants.items():
   angle=np.linalg.norm(relative_rotvec(n,value),axis=-1);out[name]=dict(root_position_rmse_m=float(np.sqrt(np.mean((value[:,:3]-n[:,:3])**2))),root_orientation_p95_rad=float(np.percentile(angle,95)),joint_position_rmse_rad=float(np.sqrt(np.mean((value[:,13:43]-n[:,13:43])**2))))
  results.append(dict(source=source,split=split,frames=len(s),variants=out))
report=dict(plan=plan,alpha_raw=raw,alpha=alphas,fit_frames=sum(len(r['data']['state']) for r in fit),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),sources=[{k:v for k,v in r.items() if k!='data'} for r in records],results=results,promoted=False)
(a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(alpha=alphas,results=results)))
