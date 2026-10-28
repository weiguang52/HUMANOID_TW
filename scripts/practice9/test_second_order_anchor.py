"""Loss and reset contracts for an inactive second-order mean-action prototype."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from second_order_anchor import PreviousTwoPoliciesTraining,second_order_anchor_mse,PREVIOUS,OLDER,VALID

# Arbitrarily changing teachers cancel. Constant-slope residual correction has
# zero curvature; rapid alternating residuals do not.
torch.manual_seed(59);teachers=[torch.randn(3,4) for _ in range(3)];slope=torch.randn(3,4);bias=torch.randn(3,4);student=[t+bias+i*slope for i,t in enumerate(teachers)];valid=torch.ones(3,1)
loss=second_order_anchor_mse(student[2],teachers[2],student[1],teachers[1],student[0],teachers[0],valid);assert loss<1e-12
amp=torch.ones(3,4);loss=second_order_anchor_mse(teachers[2]+amp,teachers[2],teachers[1]-amp,teachers[1],teachers[0]+amp,teachers[0],valid);assert torch.allclose(loss,torch.tensor(16.))
# Reset rows must contribute no gradient, including a fully masked batch.
x=torch.randn(3,4,requires_grad=True);z=torch.zeros_like(x);v=torch.tensor([[1.],[0.],[1.]])
loss=second_order_anchor_mse(x,z,z,z,z,z,v);loss.backward();assert torch.equal(x.grad[1],torch.zeros(4)) and x.grad[[0,2]].abs().sum()>0
x=torch.randn(3,4,requires_grad=True);loss=second_order_anchor_mse(x,z,z,z,z,z,torch.zeros(3,1));loss.backward();assert loss==0 and not x.grad.any()

class Fake:
 def __init__(self):self.num_envs=2;self.episode_length_buf=torch.zeros(2,dtype=torch.long);self.policy=torch.zeros(2,3);self.tick=0
 def get_observations(self):return TensorDict({'policy':self.policy,'critic':self.policy[:,:2]},batch_size=[2])
 def reset(self):self.policy.zero_();self.tick=0;self.episode_length_buf.zero_();return self.get_observations(),{}
 def step(self,actions):
  self.tick+=1;self.policy.add_(1);self.episode_length_buf.add_(1);dones=torch.tensor([False,self.tick==2])
  if dones[1]:self.policy[1]=0;self.episode_length_buf[1]=0
  return self.get_observations(),torch.zeros(2),dones,{}
env=Fake();wrap=PreviousTwoPoliciesTraining(env);o,_=wrap.reset();snap=o['policy'].clone();assert not o[VALID].any()
a,_,_,_=wrap.step(None);assert not a[VALID].any() and torch.equal(a[PREVIOUS],snap)
b,_,_,_=wrap.step(None);assert b[VALID].flatten().tolist()==[1.,0.] and torch.equal(b[OLDER],snap)
repeat=wrap.get_observations();assert torch.equal(repeat[VALID],b[VALID]) and torch.equal(repeat[OLDER],b[OLDER])
c,_,_,_=wrap.step(None);assert c[VALID].flatten().tolist()==[1.,0.]
d,_,_,_=wrap.step(None);assert d[VALID].flatten().tolist()==[1.,1.]
assert torch.equal(d[OLDER][1],torch.zeros(3)) and torch.equal(d[PREVIOUS][1],torch.ones(3))
assert torch.equal(b[OLDER],snap) # in-place env updates must not mutate snapshots
wrap.episode_length_buf=torch.tensor([8,9]);assert env.episode_length_buf.tolist()==[8,9]
print('Second-order linear residual cancellation, alternating response, masked gradients, mixed resets and immutable two-step snapshots passed')
from rsl_rl.storage import RolloutStorage
obs=TensorDict({'policy':torch.tensor([[2.,3.,4.],[12.,13.,14.]]),PREVIOUS:torch.tensor([[1.,2.,3.],[11.,12.,13.]]),OLDER:torch.tensor([[0.,1.,2.],[10.,11.,12.]]),VALID:torch.ones(2,1)},batch_size=[2]);storage=RolloutStorage('rl',2,3,obs,[2])
for step in range(3):
 tr=RolloutStorage.Transition();tr.observations=obs.clone();tr.actions=torch.zeros(2,2);tr.rewards=torch.zeros(2);tr.dones=torch.zeros(2);tr.values=torch.zeros(2,1);tr.actions_log_prob=torch.zeros(2);tr.distribution_params=(torch.zeros(2,2),torch.ones(2,2));tr.hidden_states=(None,None);storage.add_transition(tr)
 for k in ['policy',PREVIOUS,OLDER]:obs[k]=obs[k]+10
for batch in storage.mini_batch_generator(3,1):
 assert torch.equal(batch.observations['policy']-1,batch.observations[PREVIOUS]);assert torch.equal(batch.observations['policy']-2,batch.observations[OLDER]);assert batch.observations[VALID].all()
# Actual Gaussian MLP forwards: auxiliary inputs do not change actor inputs,
# deterministic prior forwards preserve cached PPO distribution and RNG.
from rsl_rl.models import MLPModel
import copy
torch.manual_seed(156);o=TensorDict({'policy':torch.randn(32,4)},batch_size=[32]);actor=MLPModel(o,{'actor':['policy']},'actor',2,hidden_dims=[8],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5});teacher=copy.deepcopy(actor).eval().requires_grad_(False)
with torch.no_grad():initial=actor(o).clone()
o[PREVIOUS]=o['policy']*.5-.3;o[OLDER]=o['policy']*.1+.4;o[VALID]=torch.ones(32,1)
with torch.no_grad():assert torch.equal(actor(o),initial)
p=o.clone(False);p['policy']=o[PREVIOUS];q=o.clone(False);q['policy']=o[OLDER]
with torch.no_grad():
 actor(o,stochastic_output=True);cached=actor.output_mean.clone();rng=torch.random.get_rng_state();actor(p);actor(q);assert torch.equal(actor.output_mean,cached) and torch.equal(torch.random.get_rng_state(),rng)
 for v in actor.mlp.parameters():v.add_(.1)
 frozen={k:v.clone() for k,v in teacher.state_dict().items()}
def actual_loss():return second_order_anchor_mse(actor(o),teacher(o),actor(p),teacher(p),actor(q),teacher(q),o[VALID])
optimizer=torch.optim.SGD(actor.parameters(),lr=.01);before=actual_loss().item();optimizer.zero_grad();loss=actual_loss();loss.backward();assert any(v.grad is not None and v.grad.abs().sum()>0 for v in actor.mlp.parameters());optimizer.step();after=actual_loss().item();assert 0<after<before
assert all(v.grad is None for v in teacher.parameters()) and all(torch.equal(v,teacher.state_dict()[k]) for k,v in frozen.items())
print('Real shuffled rollout triplets, unchanged actor input, Gaussian cache/RNG, frozen teacher and model gradient passed',before,after)
