"""Read-only verification of raw versus training-coordinate model export."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact',type=Path,required=True)
p.add_argument('--collection',type=Path,required=True)
p.add_argument('--branches',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
c=torch.load(a.artifact,map_location='cpu',weights_only=False)
plan=c['calibration_plan'];assert plan['quaternion_conditioning'] is True
assert 'optimizer' not in c
assert c['optimizer_coordinate_contract'].startswith('Conditioned quaternion')
cfg=c['cfg'];w=c['world_model']
m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,
 goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),
 termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
m.load_state_dict(w);m.eval();conditioned=copy.deepcopy(m)
conditioned.net[0].load_state_dict(c['training_coordinate_first_linear'])
mean=torch.tensor(plan['quaternion_mean'],device='cuda:0')
std=torch.tensor(plan['quaternion_std'],device='cuda:0')
def hook(module,args):
 x=args[0].clone();x[...,3:7]=(x[...,3:7]-mean)/std
 return (x,)
conditioned.net[0].register_forward_pre_hook(hook)
fit=json.loads((a.artifact.parent/'fit_input.json').read_text())
rows=[];maximum=0.;count=0
with torch.no_grad():
 for root,key in [(a.collection,'trajectory_files'),(a.branches,'one_step_files')]:
  for source in fit[key]:
   path=(root/source['file']).resolve();assert path.is_relative_to(root.resolve())
   assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
   with np.load(path,allow_pickle=False) as z:
    arrays=[torch.tensor(z[k],device='cuda:0') for k in ['state','action','context']]
   error=0.
   for start in range(0,len(arrays[0]),256):
    inputs=[v[start:start+256] for v in arrays]
    for _ in range(5):m(*inputs);conditioned(*inputs)
    error=max(error,float((m(*inputs)-conditioned(*inputs)).abs().max()))
   rows.append(dict(file=str(path),frames=len(arrays[0]),max_error=error))
   count+=len(arrays[0]);maximum=max(maximum,error)
assert count==4505
result=dict(passed=maximum<=1e-5,frames=count,max_error=maximum,atol=1e-5,
 artifact_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),rows=rows,
 scope='All original fit-state forward predictions; not held-out accuracy or policy acceptance')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
assert result['passed']
