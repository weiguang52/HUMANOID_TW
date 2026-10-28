"""Current gate survives async resets while physical action pipeline stays exact."""
import pathlib,sys,types
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from controlled_action_likelihood import ACTIVE,ControlledActionActivity
from candidate_action_scope import EXPECTED_NAMES,make_scope_contract
from training_speed_blend import TrainingSpeedBlend
from training_action_filter import VectorAnkleFilter
names=list(EXPECTED_NAMES)
class Teacher(torch.nn.Module):
 def forward(self,p):return p[:,:30]*.2

for device in ['cpu','cuda']:
 if device=='cuda' and not torch.cuda.is_available():continue
 class Fake:
  def __init__(self):
   self.num_envs=8;self.device=device;self.i=0
   self.episode_length_buf=torch.zeros(8,device=device)
   self.command=types.SimpleNamespace(cfg=types.SimpleNamespace(body_names=['base_link']),body_lin_vel_w=torch.zeros(8,1,3,device=device))
   self.filter=VectorAnkleFilter(names,8,device,.5)
   self.obs=TensorDict({'policy':torch.arange(8*203,device=device,dtype=torch.float32).reshape(8,203)/1000},batch_size=[8])
   self.obs['contact_base_policy']=self.obs['policy'][:,:195].clone();self.set_speed()
  def set_speed(self):
   self.command.body_lin_vel_w[:,0,0]=torch.tensor([0.,.04,.05,.05000001,.075,.1,.12,.2],device=device).roll(self.i%8)
  def get_observations(self):return self.obs
  def reset(self):
   self.i=0;self.set_speed();self.filter.reset(torch.ones(8,device=device,dtype=torch.bool));return self.obs,{}
  def step(self,a):
   self.last=self.filter(a).clone();self.i+=1
   done=torch.tensor([self.i%(i+3)==0 for i in range(8)],device=device)
   self.filter.reset(done);self.obs=self.obs.clone();self.obs['policy']+=.0001
   self.obs['contact_base_policy']=self.obs['policy'][:,:195].clone();self.set_speed()
   return self.obs,torch.zeros(8,device=device),done,{}
 left=Fake();right=Fake();activity=ControlledActionActivity(right,right.command)
 c={'version':2,'candidate_action_scope':make_scope_contract(names,'lower_body_waist')}
 old=TrainingSpeedBlend(left,Teacher(),left.command,c,0,action_names=names)
 new=TrainingSpeedBlend(activity,Teacher(),right.command,c,0,action_names=names)
 oldobs=old.get_observations();newobs=new.get_observations()
 rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 for step in range(256):
  raw=(torch.arange(240,device=device,dtype=torch.float32).reshape(8,30)+step)/100;raw_before=raw.clone()
  assert torch.equal(newobs[ACTIVE][:,0].bool(),activity._weight()>0)
  assert newobs[ACTIVE].dtype==torch.float32
  oldobs,lr,ld,le=old.step(raw);newobs,nr,nd,ne=new.step(raw)
  assert torch.equal(left.last,right.last) and torch.equal(ld,nd) and torch.equal(lr,nr)
  assert torch.equal(oldobs['policy'],newobs['policy']) and torch.equal(raw,raw_before)
  assert ACTIVE not in right.obs
  if step%19==0:
   repeated=new.get_observations();assert torch.equal(newobs[ACTIVE],repeated[ACTIVE])
 assert activity.completed_steps==256 and len(activity.trace)==64
 assert torch.equal(rng,torch.random.get_rng_state())
 if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
 new.reset();assert torch.equal(activity.current_active[:,0],activity._weight()>0)
 new.episode_length_buf=torch.ones(8,device=device);assert torch.equal(right.episode_length_buf,torch.ones(8,device=device))
 right.command.body_lin_vel_w.fill_(.2)
 try:new.step(torch.zeros(8,30,device=device))
 except RuntimeError:pass
 else:raise AssertionError('Stale gate accepted')
 print(device,'256-step exact physical action/filter parity, current gate, repeated get, async resets, stale guard, input/RNG passed')
