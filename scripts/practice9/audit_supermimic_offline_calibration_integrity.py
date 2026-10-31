"""Verify provenance and frozen tensors of a non-policy offline dynamics artifact."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser(description=__doc__)
for key in ['artifact','source-checkpoint','collection','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
c=torch.load(a.source_checkpoint,map_location='cpu',weights_only=False);d=torch.load(a.artifact,map_location='cpu',weights_only=False)
assert d['kind'] in ['offline_dynamics_calibration_not_policy_checkpoint','offline_pose_velocity_calibration_not_policy_checkpoint','offline_sole_geometry_dynamics_not_policy_checkpoint'] and d['calibration_updates']==1000
assert d['calibration_plan']['source_checkpoint_sha256']==hashlib.sha256(a.source_checkpoint.read_bytes()).hexdigest()
if d['kind'] in ['offline_pose_velocity_calibration_not_policy_checkpoint','offline_sole_geometry_dynamics_not_policy_checkpoint']:
 assert d['prediction_contract']==d['calibration_plan']['prediction_contract']
 assert d['prediction_contract']['contract']=='pose_from_velocity_fit_alpha_v1'
assert not any(k in d for k in ['actor_critic','actor_optimizer','critic_optimizer'])
count=0
def finite(x):
 global count
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();count+=1
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
finite(d)
if d['kind']=='offline_sole_geometry_dynamics_not_policy_checkpoint':
 from supermimic_geometry_model import load_geometry_artifact,install_geometry_input
 model,_=load_geometry_artifact(d,device='cpu')
 from supermimic.algo.networks import WorldModel,WorldModelCfg
 cfg=c['cfg'];w=c['world_model']
 parent=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims'])))
 parent.load_state_dict(w);parent=parent.cuda();install_geometry_input(parent,d['geometry_contract'],cfg)
 c=dict(c,world_model={k:v.cpu() for k,v in parent.state_dict().items()})
 for key,value in d['world_model'].items():
  if key.startswith('net.0.') and key not in ['net.0.weight','net.0.bias']:assert torch.equal(value,c['world_model'][key]),key
 assert torch.count_nonzero(d['world_model']['net.0.weight'][:,:2])==0
 assert not d['smoke_only'] and d['geometry_contract']==d['calibration_plan']['geometry_contract']
assert d['world_model'].keys()==c['world_model'].keys()
frozen=[];changed=[]
for k,v in d['world_model'].items():
 if not k.startswith('net.'):
  assert torch.equal(v,c['world_model'][k]),k;frozen.append(k)
 elif not torch.equal(v,c['world_model'][k]):changed.append(k)
assert changed
mapping=json.loads((a.collection/'control_mapping.json').read_text())
for mk,ck in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[mk]==c['cfg']['extra'][ck]
rows=json.loads((a.collection/'results.json').read_text())['rows'];fit=[r['trace'] for r in rows if r['split']=='fit'];other={r['trace'] for r in rows if r['split']!='fit'}
assert len(set(fit))==len(fit) and not set(fit)&other
inputs=json.loads((a.artifact.parent/'fit_input.json').read_text());assert inputs['traces']==fit
window=d['calibration_plan']['window'];windows=0;files=[]
for name in fit:
 path=(a.collection/name).resolve();assert path.is_relative_to(a.collection.resolve())
 with np.load(path,allow_pickle=False) as z:
  n=len(z['state']);assert np.array_equal(z['next_state'][:-1],z['state'][1:])
  for k,width in [('state',73),('next_state',73),('action',30),('context',34)]:assert z[k].shape==(n,width) and np.isfinite(z[k]).all()
  windows+=max(0,n-window+1)
 files.append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert windows==inputs['windows']
report=dict(passed=True,artifact_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),finite_tensor_count=count,window=window,fit_windows=windows,fit_files=files,frozen_nonnet_tensors=frozen,changed_dynamics_tensors=changed,policy_artifact=False,promoted=False)
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['fit_files','frozen_nonnet_tensors','changed_dynamics_tensors']}))
