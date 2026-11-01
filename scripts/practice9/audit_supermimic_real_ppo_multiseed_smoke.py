"""Seed42 legacy regression, new training seeds and reused fixed-budget evidence."""
import argparse,json,hashlib
from pathlib import Path
import torch
from audit_supermimic_real_ppo_artifact import audit
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root;s=r/'real_ppo_multiseed_standing_v1'
parent=Path('/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt');manifest=r.parent/'static_probe/manifest.json'
reports=[]
for label,seed in [('legacy42',42),('default42',42),('seed43',43),('smooth43',43)]:
 reports.append(dict(label=label,audit=audit(s/'smoke'/label,parent,manifest,4,8,seed)))
def same(x,y):
 if torch.is_tensor(x):return torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
 return x==y
load=lambda label:torch.load(s/'smoke'/label/'checkpoint_final.pt',map_location='cpu',weights_only=False)
old,new,other=[load(x) for x in ['legacy42','default42','seed43']]
assert old['plan']==new['plan'] and old['cfg']==new['cfg']
for key in ['actor_critic','world_model','ppo_optimizer','torch_rng_state','cuda_rng_states']:assert same(old[key],new[key]),key
assert not same(new['actor_critic'],other['actor_critic']) and not same(new['torch_rng_state'],other['torch_rng_state'])
assert new['plan']['seed']==42 and other['plan']['seed']==43
reused=[]
for arm,relative,coefficient,loss in [('baseline','real_ppo_standing_v1',0.,'squared'),('smooth','sole_distance_ppo_v1/trained',2.,'smooth_distance')]:
 run=r/relative;report=audit(run,parent,manifest,1000,256,42)
 c=torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False)
 assert c['plan'].get('sole_normal_penalty',0.)==coefficient and c['plan'].get('sole_normal_loss','squared')==loss
 reused.append(dict(arm=arm,seed=42,path=str(run),sha256=hashlib.sha256((run/'checkpoint_final.pt').read_bytes()).hexdigest(),audit=report))
out=dict(passed=True,default_seed42_actor_world_optimizer_rng_bit_identical_to_legacy=True,seed43_training_and_rng_different=True,smoke=reports,reused_seed42=reused,scope='Artifact/seed/adapter checks only. No standing, multi-seed efficacy, full-motion or policy promotion acceptance.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['smoke','reused_seed42']}))
