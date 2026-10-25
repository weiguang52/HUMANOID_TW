"""Tiny CPU PPO tests: parity, anchoring gradient, teacher and RNG invariants."""
import copy,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from anchored_ppo import install_anchor

def make():
 obs=TensorDict({'policy':torch.randn(32,4)},batch_size=[32])
 groups={'actor':['policy'],'critic':['policy']}
 actor=MLPModel(obs,groups,'actor',2,hidden_dims=[8],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5})
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=[8])
 with torch.no_grad():
  actions=actor(obs,stochastic_output=True)
  batch=SimpleNamespace(observations=obs,actions=actions,old_actions_log_prob=actor.get_output_log_prob(actions).unsqueeze(-1),values=critic(obs),advantages=torch.randn(32,1),returns=torch.randn(32,1),old_distribution_params=tuple(x.clone() for x in actor.output_distribution_params),masks=None,hidden_states=(None,None))
 class Storage:
  def mini_batch_generator(self,*args):yield copy.deepcopy(batch)
  def clear(self):pass
 return PPO(actor,critic,Storage(),num_learning_epochs=1,num_mini_batches=1,schedule='fixed',learning_rate=.001),obs,batch

torch.manual_seed(17)
base,obs,batch=make();other=copy.deepcopy(base)
identity=type(other);assert not install_anchor(other,0)['enabled'] and type(other) is identity
install_anchor(other,1.);other.anchor_coefficient=0.
rng=torch.random.get_rng_state();base.update();torch.random.set_rng_state(rng);other.update()
for a,b in zip(base.actor.parameters(),other.actor.parameters()):assert torch.equal(a,b)
for a,b in zip(base.critic.parameters(),other.critic.parameters()):assert torch.equal(a,b)
print('zero-anchor copied-update parity passed')
alg,obs,batch=make();install_anchor(alg,1.)
assert not any(p.requires_grad for p in alg.anchor_teacher.parameters())
state={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()}
with torch.no_grad():
 for p in alg.actor.mlp.parameters():p.add_(.1)
 teacher=alg.anchor_teacher(obs)
 before=((alg.actor(obs)-teacher)**2).mean().item()
# Disable all other losses to test the sign and real optimizer participation.
batch.advantages.zero_();alg.value_loss_coef=0.;alg.entropy_coef=0.
losses=alg.update()
with torch.no_grad():after=((alg.actor(obs)-teacher)**2).mean().item()
assert after<before and losses['anchor_mse']>0
assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in state.items())
assert all(p.grad is None for p in alg.anchor_teacher.parameters())
rng=torch.random.get_rng_state();alg.anchor_teacher(obs);assert torch.equal(rng,torch.random.get_rng_state())
for bad in [-1.,float('nan')]:
 try:install_anchor(alg,bad)
 except ValueError:pass
 else:raise AssertionError('invalid coefficient accepted')
print('anchor gradient, frozen teacher, deterministic RNG and validation passed',before,after)
fresh,obs,batch=make();original={k:v.clone() for k,v in fresh.actor.state_dict().items()}
with torch.no_grad():
 for p in fresh.actor.parameters():p.add_(.2)
install_anchor(fresh,1.,original)
assert all(torch.equal(v,fresh.anchor_teacher.state_dict()[k]) for k,v in original.items())
assert any(not torch.equal(v,fresh.actor.state_dict()[k]) for k,v in original.items())
print('explicit original teacher survives student checkpoint changes')
