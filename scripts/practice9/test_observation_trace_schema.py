import sys,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import numpy as np,torch
from policy_observation_trace import PolicyObservationTrace
manager=SimpleNamespace(active_terms={'policy':['position','velocity']},group_obs_term_dim={'policy':[(2,),(3,)]},group_obs_dim={'policy':(5,)})
env=SimpleNamespace(unwrapped=SimpleNamespace(observation_manager=manager))
x={'policy':torch.ones(1,7),'contact_base_policy':torch.ones(1,5)};a=torch.ones(1,3)
t=PolicyObservationTrace(env);t.append(x,a);x['policy'].zero_();a.zero_()
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'trace.npz';t.save(p);z=np.load(p)
 assert z['base_policy_term_names'].tolist()==['position','velocity']
 assert z['base_policy_term_slices'].tolist()==[[0,2],[2,5]]
 assert int(z['base_policy_dimension'])==5 and np.all(z['action']==1)
 assert np.all(z['observation_1']==1)
 legacy=PolicyObservationTrace();legacy.append({'policy':torch.ones(1,5)},torch.ones(1,3));legacy.save(p)
 assert not any(k.startswith('base_policy_') for k in np.load(p).files)
try:t.append({'policy':torch.ones(1,7),'contact_base_policy':torch.ones(1,4)},torch.ones(1,3))
except ValueError:pass
else:raise AssertionError('Mismatched base width accepted')
manager.group_obs_dim['policy']=(6,)
try:PolicyObservationTrace(env)
except ValueError:pass
else:raise AssertionError('Mismatched live dimensions accepted')
print('Live schema slices, immutable snapshots, extended/base widths, malformed schema and unchanged legacy format passed')
