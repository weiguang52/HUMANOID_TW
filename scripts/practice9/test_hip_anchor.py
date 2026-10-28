"""Targeted anchor gradient and real PPO parity checks."""
import sys,runpy,copy
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from anchored_ppo import anchor_mse,hip_anchor_weights,install_anchor,HIP_ANCHOR_NAMES
ns=runpy.run_path(str(Path(__file__).with_name('test_anchored_ppo.py')))
make=ns['make']
names=['arm_a',HIP_ANCHOR_NAMES[2],HIP_ANCHOR_NAMES[0],'arm_b',HIP_ANCHOR_NAMES[3],HIP_ANCHOR_NAMES[1]]
w=hip_anchor_weights(names,3)
assert w==[1,3,3,1,3,3]
x=torch.tensor([[1.,2.,3.,4.,5.,6.]],requires_grad=True)
t=torch.zeros_like(x)
a=anchor_mse(x,t);g=torch.autograd.grad(a,x,retain_graph=True)[0]
b=anchor_mse(x,t,torch.tensor(w));h=torch.autograd.grad(b,x)[0]
assert torch.allclose(h,g*torch.tensor(w))
assert torch.equal(anchor_mse(x,t),anchor_mse(x,t,torch.ones(6)))
base,obs,batch=make();weighted=copy.deepcopy(base)
install_anchor(base,1.);install_anchor(weighted,1.,action_weights=[1.,1.])
with torch.no_grad():
 for p,q in zip(base.actor.parameters(),weighted.actor.parameters()):
  p.add_(.1);q.add_(.1)
rng=torch.random.get_rng_state();base.update();torch.random.set_rng_state(rng);weighted.update()
assert all(torch.equal(a,b) for a,b in zip(base.actor.parameters(),weighted.actor.parameters()))
assert all(torch.equal(a,b) for a,b in zip(base.critic.parameters(),weighted.critic.parameters()))
for bad in [float('nan'),0.,-1.,float('inf')]:
 try:hip_anchor_weights(names,bad)
 except ValueError:pass
 else:raise AssertionError('invalid multiplier accepted')
for badnames in [names[:-1],names+['arm_a']]:
 try:hip_anchor_weights(badnames,3)
 except ValueError:pass
 else:raise AssertionError('invalid names accepted')
for bad in [[0,1],[-1,1],[float('nan'),1],[],[[1,1]]]:
 alg,_,_=make()
 try:install_anchor(alg,1.,action_weights=bad)
 except ValueError:pass
 else:raise AssertionError('invalid weights accepted')
try:anchor_mse(x,t,torch.ones(2))
except ValueError:pass
else:raise AssertionError('dimension mismatch accepted')
alg,obs,batch=make();install_anchor(alg,1.,action_weights=[3.,1.])
with torch.no_grad():
 for p in alg.actor.mlp.parameters():p.add_(.1)
 expected=anchor_mse(alg.actor(obs),alg.anchor_teacher(obs),alg.anchor_action_weights).item()
batch.advantages.zero_();alg.value_loss_coef=0.;alg.entropy_coef=0.
state={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()}
losses=alg.update()
assert abs(losses['anchor_mse']-expected)<1e-7
with torch.no_grad():after=anchor_mse(alg.actor(obs),alg.anchor_teacher(obs),alg.anchor_action_weights).item()
assert after<expected
assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in state.items())
assert all(p.grad is None for p in alg.anchor_teacher.parameters())
print('real weighted PPO loss and optimizer participation passed',expected,after)
print('hip anchor mapping, gradient ratios, default/all-one PPO parity and rejection checks passed')
