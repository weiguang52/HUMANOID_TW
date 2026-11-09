"""Audit checkpoint identity, optimizer state and unchanged training sources."""
import json,hashlib,subprocess,re
from pathlib import Path
import torch
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')
BASE='2e1a2ae536018641c3efd05680874e7e14910686'

def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();return 1
 if isinstance(x,dict):return sum(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)):return sum(finite(v) for v in x)
 return 0

def audit():
 models=json.loads((ROOT/'input_models.json').read_text());audits={};sources={}
 baseline=torch.load(models['requested_s1_u1500']['checkpoint'],map_location='cpu',weights_only=False)
 def shapes(m):return {g+'/'+k:list(v.shape) for g in ['actor_state_dict','critic_state_dict'] for k,v in m[g].items() if isinstance(v,torch.Tensor)}
 expected_shapes=shapes(baseline)
 for name,e in models.items():
  p=Path(e['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'];m=torch.load(p,map_location='cpu',weights_only=False);assert m['iter']==e['iteration']
  assert 'optimizer_state_dict' in m and m['optimizer_state_dict']['state']
  assert 'actor_state_dict' in m and 'critic_state_dict' in m
  assert shapes(m)==expected_shapes
  steps=[float(v['step']) for v in m['optimizer_state_dict']['state'].values() if 'step' in v];assert steps and min(steps)>0
  audits[name]={'sha256':e['sha256'],'iteration':m['iter'],'finite_tensors':finite(m),'optimizer_state_present':True,'optimizer_step_min':min(steps),'optimizer_step_max':max(steps),'actor_tensor_count':len(m['actor_state_dict']),'critic_tensor_count':len(m['critic_state_dict']),'architecture_matches_requested_baseline':True}
 for name in ['train.py','eval_cmd.py','remote_env.sh']:
  p=Path('scripts/locomotion')/name;h=hashlib.sha256(p.read_bytes()).hexdigest();old=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+str(p)])).hexdigest();assert h==old;sources[str(p)]={'current':h,'base':old}
 old_sources=json.loads(Path('validation_artifacts/stage1_issue9_20261109/training_source_identity.json').read_text())
 for name in old_sources['sha256']:
  p=Path(name);h=hashlib.sha256(p.read_bytes()).hexdigest();old=hashlib.sha256(subprocess.check_output(['git','show',BASE+':'+str(p)])).hexdigest();assert h==old;sources[str(p)]={'current':h,'base':old}
 frozen=json.loads((ROOT/'frozen_plan_sha256.json').read_text());assert hashlib.sha256((ROOT/'frozen_plan.md').read_bytes()).hexdigest()==frozen['frozen_plan.md']
 (ROOT/'input_checkpoint_audit.json').write_text(json.dumps({'passed':True,'checkpoints':audits,'unchanged_sources':sources},indent=2));print('INPUT_AUDIT_PASS',len(audits))
if __name__=='__main__':audit()
