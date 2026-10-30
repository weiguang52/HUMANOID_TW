"""Bounded fit-only dynamics calibration; actor, termination head and norms frozen."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
plan=dict(seed=42,updates=1000,batch=128,window=8,lr=.0001,optimizer='fresh Adam on dynamics net only',frozen=['actor','critic','termination head','normalizers'],selection='Final update1000 only; no best checkpoint',diagnostic_gate='Validation episode-median root position and orientation p95 improve >=20% at H1/H8/H32; no other reported RMSE median worsens >5%. Diagnostic only, not skill acceptance.',source_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');torch.manual_seed(42)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg'];m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w)
for v in m.parameters():v.requires_grad_(False)
for v in m.net.parameters():v.requires_grad_(True)
opt=torch.optim.Adam(m.net.parameters(),lr=plan['lr']);states=[];actions=[];contexts=[];targets=[];used=[]
for row in json.loads((a.collection/'results.json').read_text())['rows']:
 if row['split']!='fit':continue
 with np.load(a.collection/row['trace']) as d:
  for start in range(len(d['state'])-8+1):
   states.append(d['state'][start]);actions.append(d['action'][start:start+8]);contexts.append(d['context'][start:start+8]);targets.append(d['next_state'][start:start+8])
 used.append(row['trace'])
s,act,ctx,target=[torch.tensor(np.stack(x),device='cuda:0') for x in (states,actions,contexts,targets)];assert len(s)>128
(a.output/'fit_input.json').write_text(json.dumps(dict(traces=used,windows=len(s)),indent=2))
for update in range(1,1001):
 idx=torch.randint(len(s),(128,),device='cuda:0');pred=s[idx];loss=0
 for j in range(8):pred=m(pred,act[idx,j],ctx[idx,j]);loss=loss+m.state_loss(pred,target[idx,j])/8
 assert torch.isfinite(loss);opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.net.parameters(),cfg['wm_grad_clip']);assert torch.isfinite(norm);opt.step()
 if update%100==0:print(json.dumps(dict(update=update,loss=float(loss),gradient_norm=float(norm))),flush=True)
for key,value in m.state_dict().items():
 if not key.startswith('net.'):assert torch.equal(value.cpu(),w[key]),key
artifact=dict(kind='offline_dynamics_calibration_not_policy_checkpoint',world_model=m.state_dict(),cfg=cfg,calibration_plan=plan,calibration_updates=1000,optimizer=opt.state_dict())
torch.save(artifact,a.output/'world_model_calibrated.pt');print('CALIBRATION_COMPLETE',flush=True)
