"""Temporal loss, episode isolation, real rollout pairing and PPO integration."""
import sys,runpy
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from temporal_anchor import PreviousPolicyTraining,PREVIOUS_POLICY,PREVIOUS_VALID,temporal_anchor_mse
from anchored_ppo import install_anchor

class Fake:
    num_envs=2
    episode_length_buf=torch.zeros(2,dtype=torch.long)
    def __init__(self):self.p=torch.tensor([[1.,2.],[11.,12.]]);self.dones=torch.tensor([1,0])
    def get_observations(self):return TensorDict({'policy':self.p,'critic':self.p+100},[2])
    def reset(self):return self.get_observations(),{}
    def step(self,actions):
        self.p.add_(1)
        self.p[self.dones.bool()]=0
        return self.get_observations(),torch.zeros(2),self.dones,{}
f=Fake();w=PreviousPolicyTraining(f)
a=w.get_observations();initial=a['policy'].clone();assert not a[PREVIOUS_VALID].any()
assert torch.equal(a['policy'],f.p) and PREVIOUS_POLICY not in f.get_observations()
assert torch.equal(w.get_observations()[PREVIOUS_VALID],a[PREVIOUS_VALID])
old=f.p.clone();b,_,_,_=w.step(None)
assert torch.equal(b[PREVIOUS_POLICY],old)
assert torch.equal(b[PREVIOUS_VALID],torch.tensor([[0.],[1.]]))
f.dones=torch.zeros(2,dtype=torch.long);old=f.p.clone();c,_,_,_=w.step(None)
assert torch.equal(c[PREVIOUS_POLICY],old) and c[PREVIOUS_VALID].all()
assert torch.equal(b[PREVIOUS_POLICY],initial)
w.episode_length_buf=torch.ones(2,dtype=torch.long);assert f.episode_length_buf.sum()==2
assert not w.reset()[0][PREVIOUS_VALID].any()

now=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True);prev=torch.zeros_like(now,requires_grad=True)
t=torch.zeros_like(now);valid=torch.tensor([[1.],[0.]])
loss=temporal_anchor_mse(now,t,prev,t,valid);gn,gp=torch.autograd.grad(loss,(now,prev))
assert torch.equal(gn[1],torch.zeros(2)) and torch.equal(gp,-gn)
zero=temporal_anchor_mse(now,t,prev,t,torch.zeros_like(valid));assert zero.item()==0
assert not torch.autograd.grad(zero,now)[0].any()
# Preserve arbitrarily changing teacher action when the residual is constant.
tn=torch.tensor([[1.,3.],[4.,2.]],dtype=torch.float64);tp=-tn
assert temporal_anchor_mse(tn+.25,tn,tp+.25,tp,torch.ones(2,1)).item()==0

# Real storage add_transition and shuffled mini-batches preserve pairing.
obs=TensorDict({'policy':torch.tensor([[1.,2.],[3.,4.]]),PREVIOUS_POLICY:torch.tensor([[0.,1.],[2.,3.]]),PREVIOUS_VALID:torch.ones(2,1)},[2])
storage=RolloutStorage('rl',2,2,obs,[2])
for step in range(2):
    tr=RolloutStorage.Transition();tr.observations=obs.clone()
    tr.actions=torch.zeros(2,2);tr.rewards=torch.zeros(2);tr.dones=torch.zeros(2)
    tr.values=torch.zeros(2,1);tr.actions_log_prob=torch.zeros(2)
    tr.distribution_params=(torch.zeros(2,2),torch.ones(2,2));tr.hidden_states=(None,None)
    storage.add_transition(tr)
    obs['policy']=obs['policy']+10;obs[PREVIOUS_POLICY]=obs[PREVIOUS_POLICY]+10
for batch in storage.mini_batch_generator(2,1):
    assert torch.equal(batch.observations['policy']-1,batch.observations[PREVIOUS_POLICY])
    assert batch.observations[PREVIOUS_VALID].all()

ns=runpy.run_path(str(Path(__file__).with_name('test_anchored_ppo.py')))
alg,obs,batch=ns['make']();assert alg.actor.obs_groups==['policy']
with torch.no_grad():base_actor=alg.actor(obs).clone();base_critic=alg.critic(obs).clone()
obs[PREVIOUS_POLICY]=obs['policy']*.4-.2
with torch.no_grad():
    assert torch.equal(base_actor,alg.actor(obs)) and torch.equal(base_critic,alg.critic(obs))
obs[PREVIOUS_VALID]=torch.ones(32,1);obs[PREVIOUS_VALID][:3]=0
install_anchor(alg,0.,temporal_coefficient=1.)
with torch.no_grad():
    for p in alg.actor.mlp.parameters():p.add_(.1)
    current=alg.actor(obs)
    previous_obs=obs.clone(False);previous_obs['policy']=obs[PREVIOUS_POLICY]
    alg.actor(obs,stochastic_output=True)
    cached=alg.actor.output_mean.clone();rng=torch.random.get_rng_state()
    previous=alg.actor(previous_obs)
    assert torch.equal(cached,alg.actor.output_mean)
    assert torch.equal(rng,torch.random.get_rng_state())
    expected=temporal_anchor_mse(current,alg.anchor_teacher(obs),previous,alg.anchor_teacher(previous_obs),obs[PREVIOUS_VALID]).item()
state={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()}
batch.advantages.zero_();alg.value_loss_coef=0.;alg.entropy_coef=0.
losses=alg.update();assert abs(losses['anchor_temporal_mse']-expected)<1e-7
with torch.no_grad():
    after=temporal_anchor_mse(alg.actor(obs),alg.anchor_teacher(obs),alg.actor(previous_obs),alg.anchor_teacher(previous_obs),obs[PREVIOUS_VALID]).item()
assert after<expected
assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in state.items())
assert all(p.grad is None for p in alg.anchor_teacher.parameters())
for bad in [-1.,float('nan'),float('inf')]:
    fresh,_,_=ns['make']()
    try:install_anchor(fresh,1.,temporal_coefficient=bad)
    except ValueError:pass
    else:raise AssertionError('invalid temporal coefficient accepted')
print('Temporal reset/mask, nominal-motion cancellation, real storage pairing and PPO gradient tests passed',expected,after)
