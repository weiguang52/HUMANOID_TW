"""Real PPO/Adam migration and reset wrapper contracts; no Isaac dependency."""
import copy,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
import contact_policy as cp
from anchored_ppo import install_anchor

class Fake:
 def __init__(self):
  self.num_envs=2;self.unwrapped=self;self.episode_length_buf=torch.zeros(2,dtype=torch.long)
  self.obs=TensorDict({'policy':torch.ones(2,4),'critic':torch.ones(2,3)},batch_size=[2])
 def get_observations(self):return self.obs
 def reset(self):self.episode_length_buf.zero_();return self.obs,{'reset':True}
 def step(self,actions):self.obs['policy'].add_(1);self.episode_length_buf.add_(1);return self.obs,torch.zeros(2),torch.tensor([False,True]),{}
env=Fake();wrap=cp.ContactPolicyObservations(env,'zeros');a=wrap.get_observations();assert a['policy'].shape==(2,12) and torch.equal(a['policy'][:,:4],a[cp.BASE_POLICY]) and not a['policy'][:,4:].any()
obs,_,done,_=wrap.step(torch.zeros(2,2));assert done.tolist()==[False,True] and torch.equal(a[cp.BASE_POLICY],torch.ones(2,4))
assert torch.equal(obs['critic'],env.obs['critic']);wrap.episode_length_buf=torch.tensor([3,4]);assert env.episode_length_buf.tolist()==[3,4]
obs,info=wrap.reset();assert info['reset'] and not env.episode_length_buf.any()
original=cp.ground_contact_features
cp.ground_contact_features=lambda env,prefix:torch.arange(16).reshape(2,8).float()
actual=cp.ContactPolicyObservations(Fake(),'actual',sensor_prefix='diagnostic_').get_observations();assert torch.equal(actual['policy'][:,4:],torch.arange(16).reshape(2,8).float());cp.ground_contact_features=original
try:wrap._pack(a)
except ValueError:pass
else:raise AssertionError('Double packing accepted')

class Storage:
 def __init__(self):self.batch=None
 def mini_batch_generator(self,*args):yield copy.deepcopy(self.batch)
 def clear(self):pass

def make(obs):
 groups={'actor':['policy'],'critic':['critic']}
 actor=MLPModel(obs,groups,'actor',2,hidden_dims=[8],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5})
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=[8]);storage=Storage()
 return PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=1,schedule='fixed',learning_rate=.001)
def set_batch(alg,obs):
 with torch.no_grad():
  actions=alg.actor(obs,stochastic_output=True)
  alg.storage.batch=SimpleNamespace(observations=obs,actions=actions,old_actions_log_prob=alg.actor.get_output_log_prob(actions).unsqueeze(-1),values=alg.critic(obs),advantages=torch.randn(len(obs),1),returns=torch.randn(len(obs),1),old_distribution_params=tuple(x.clone() for x in alg.actor.output_distribution_params),masks=None,hidden_states=(None,None))

torch.manual_seed(59)
obs=TensorDict({'policy':torch.randn(32,4),'critic':torch.randn(32,3)},batch_size=[32]);old=make(obs);set_batch(old,obs);old.update();checkpoint=copy.deepcopy(old.save())
new_obs=obs.clone();new_obs['policy']=torch.cat([obs['policy'],torch.randn(32,8)],dim=-1);new=make(new_obs)
loaded,audit=cp.migrate_checkpoint(new,checkpoint);assert audit['expanded'] and audit['optimizer_moment_tensors_padded']==2
assert new.load(loaded,None,True)
with torch.no_grad():assert torch.allclose(new.actor(new_obs),old.actor(obs),atol=1e-7,rtol=1e-6)
key=audit['key'];param=dict(new.actor.named_parameters())[key];old_param=dict(old.actor.named_parameters())[key]
for name,value in old.optimizer.state[old_param].items():
 v=new.optimizer.state[param][name]
 if value.ndim:
  assert torch.equal(v[:,:4],value) and not v[:,4:].any()
 else:assert torch.equal(v,value)
# Teacher must ignore arbitrary added features, stay frozen, and participate in
# a real PPO update; the student's new columns receive an optimizing gradient.
teacher,_=cp.expand_actor_state(new.actor,checkpoint['actor_state_dict']);install_anchor(new,1.,teacher)
with torch.no_grad():
 baseline=new.anchor_teacher(new_obs).clone();param[:,4:].add_(.05)
assert torch.allclose(baseline,old.actor(obs),atol=1e-7,rtol=1e-6)
set_batch(new,new_obs);new.storage.batch.advantages.zero_();new.value_loss_coef=0;new.entropy_coef=0
with torch.no_grad():before=((new.actor(new_obs)-baseline)**2).mean().item();columns=param[:,4:].clone()
losses=new.update()
with torch.no_grad():after=((new.actor(new_obs)-baseline)**2).mean().item()
assert after<before and not torch.equal(columns,param[:,4:]) and losses['anchor_mse']>0
assert all(not p.requires_grad and p.grad is None for p in new.anchor_teacher.parameters())
assert torch.equal(new.anchor_teacher(new_obs),baseline)
# Re-loading an already expanded checkpoint preserves all moments and dimensions.
again,second=cp.migrate_checkpoint(new,new.save());assert not second['expanded'] and second['optimizer_moment_tensors_padded']==0
for k,v in new.actor.state_dict().items():assert torch.equal(v,again['actor_state_dict'][k])
bad=copy.deepcopy(checkpoint);bad['actor_state_dict'][key]=bad['actor_state_dict'][key][:,:3]
try:cp.migrate_checkpoint(new,bad)
except ValueError:pass
else:raise AssertionError('Unexpected original input width accepted')
print('Wrapper snapshots/reset/critic and actual-vs-zero features passed')
print('Real PPO/Adam zero-column migration, unchanged initial teacher outputs, frozen teacher, extra-column gradient and reload passed',before,after)
from policy_action_blend import PolicyActionBlend
from reference_speed_blend import ReferenceSpeedBlend
class Teacher:
 def __call__(self,x):
  assert x.shape[1]==4
  return x[:,:2]
blend=object.__new__(PolicyActionBlend);blend.audit={};blend.weight=1.;blend.mask=None;blend.teacher=Teacher()
new_obs[cp.BASE_POLICY]=obs['policy'].clone();action=torch.randn(32,2)
assert torch.equal(blend(new_obs,action),obs['policy'][:,:2])
assert torch.equal(blend(obs,action),obs['policy'][:,:2])
speed_blend=object.__new__(ReferenceSpeedBlend);speed_blend.audit={};speed_blend.base=blend;speed_blend.command=SimpleNamespace(body_lin_vel_w=torch.zeros(32,1,3));speed_blend.index=0;speed_blend.maximum=1.;speed_blend.low=.05;speed_blend.high=.1;speed_blend.hip_filter=None;speed_blend.leg_filter=None;speed_blend.trace=[]
assert torch.equal(speed_blend(new_obs,action),obs['policy'][:,:2])
assert torch.equal(speed_blend(obs,action),obs['policy'][:,:2])
contract={'version':1,'mode':'actual','base_dim':195,'extra_dim':8,'load_scale_N':100.,'speed_scale_m_s':.1,'loaded_threshold_N':5.,'reset':'zero first per-environment observation','sensor_prefix_training':'slip_','teacher_input':'original policy only'}
cp.validate_contact_contract(contract)
for key,value in [('speed_scale_m_s',1.),('extra_dim',4),('mode','unknown')]:
 invalid=dict(contract);invalid[key]=value
 try:cp.validate_contact_contract(invalid)
 except ValueError:pass
 else:raise AssertionError('Invalid input contract accepted')
print('Legacy and extended teacher blend inputs, and saved normalization contract validation passed')
