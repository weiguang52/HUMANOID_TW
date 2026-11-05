"""Verify matched moment-reset experiment before independent physical validation."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--updates',type=int,choices=[4,1000],required=True);p.add_argument('--reference-control',type=Path,required=True);a=p.parse_args()
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
paths={arm:a.root/arm/'fit' for arm in ('control','candidate')}
models={arm:torch.load(path/'world_model_calibrated.pt',map_location='cpu',weights_only=False) for arm,path in paths.items()}
ref=torch.load(a.reference_control,map_location='cpu',weights_only=False)
c=models['control'];k=models['candidate'];assert equal(c['world_model'],ref['world_model']) and equal(c['optimizer'],ref['optimizer'])
frozen=[n for n in c['world_model'] if not n.startswith('net.')];assert len(frozen)==22
assert all(torch.equal(c['world_model'][n],k['world_model'][n]) for n in frozen)
for model in models.values():
 states=list(model['optimizer']['state'].values());assert len(states)==8
 assert all(float(x['step'])==a.updates for x in states)
 assert all(torch.isfinite(x['exp_avg_sq']).all() and (x['exp_avg_sq']>=0).all() for x in states)
hist={arm:json.loads((path/'static_step_guard_history.json').read_text()) for arm,path in paths.items()}
assert hist['control'][0]==hist['candidate'][0]
for arm,path in paths.items():
 h=hist[arm];assert len(h)==a.updates and all(x['applied_nonincrease'] for x in h)
 resets=json.loads((path/'moment_reset_history.json').read_text());assert len(resets)==a.updates
 assert [x['reset_tensors'] for x in resets]==([0]*a.updates if arm=='control' else [0]+[8]*(a.updates-1))
 assert [x['static_indices'] for x in h]==[x['static_indices'] for x in hist['control']]
for filename in ('native_replay_history.json','native_h32_history.json'):
 hs=[json.loads((paths[arm]/filename).read_text()) for arm in ('control','candidate')]
 assert [[x['indices'] for x in h] for h in hs][0]==[[x['indices'] for x in h] for h in hs][1]
out=dict(passed=True,updates=a.updates,control_weights_optimizer_bitexact=True,frozen_tensors=len(frozen),adam_steps_verified=8,static_indices_equal=True,native_indices_equal=True,first_step_guard_bitexact=True,policy_acceptance=False,limitations='Does not prove physical generalization; independent gates still required.',checkpoint_sha256={arm:hashlib.sha256((path/'world_model_calibrated.pt').read_bytes()).hexdigest() for arm,path in paths.items()})
(a.root/'fit_integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
