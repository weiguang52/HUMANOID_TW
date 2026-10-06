"""Verify matching resumed policies and one-factor smoke configurations."""
import json,hashlib
from pathlib import Path
import torch,yaml
S=Path('/root/gpufree-data/datasets/practice9/tw170_v1')
base=Path((S.parent/'tw154_v1/rate/final_checkpoint').read_text().strip())
result={'base_checkpoint':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'runs':{}}
configs=[];runtimes=[]
for v in ['control','knee']:
 out=S/f'smoke_{v}'
 assert (out/'status').read_text().strip()=='trained'
 p=Path((out/'final_checkpoint').read_text().strip())
 ck=torch.load(p,map_location='cpu',weights_only=False)
 for key in ['actor_state_dict','critic_state_dict']:
  assert all(torch.isfinite(t).all() for t in ck[key].values())
 assert ck['iter']>=20002,ck['iter']
 agent=yaml.load((p.parent/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
 assert agent['resume']=='true'
 log=(out/'train.log').read_text();assert str(base) in log
 cfg=yaml.load((p.parent/'params/env.yaml').read_text(),Loader=yaml.BaseLoader);configs.append(cfg)
 runtime=json.loads((p.parent/'params/control_runtime.json').read_text());runtimes.append(runtime)
 result['runs'][v]={'checkpoint':str(p),'iteration':ck['iter'],'finite':True,'loaded_base':True}
assert runtimes[0]==runtimes[1]
assert 'swing_knee_residual' not in configs[0]['rewards']
term=configs[1]['rewards'].pop('swing_knee_residual');assert float(term['weight'])==-.5
# Both task configs otherwise serialize identically, including observations and actions.
assert configs[0]==configs[1], 'Unexpected differences beyond the one reward'
result['only_config_difference']={'swing_knee_residual':term}
print(json.dumps(result,indent=2))
