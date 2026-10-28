"""Same-state deployed/training action parity, reset, noise and contracts."""
import sys,pathlib,types,copy,tempfile,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch,numpy as np
from training_speed_blend import TrainingSpeedBlend,validate_resume,validate_jobs,validate_checkpoint_jobs,make_contract
from reference_speed_blend import ReferenceSpeedBlend
from training_action_filter import VectorAnkleFilter
names=[s+'_'+j for s in ('left','right') for j in ('hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll')]+['other'+str(i) for i in range(16)]
class Teacher(torch.nn.Module):
 def forward(self,x):return x[:,:30]*.2
class Fake:
 def __init__(self,device):
  self.num_envs=4;self.device=device;self.i=0;self.episode_length_buf=torch.zeros(4,device=device)
  self.action_filter=VectorAnkleFilter(names,4,device,.5)
  self.obs={'policy':torch.arange(4*203,device=device,dtype=torch.float32).reshape(4,203)/1000}
  self.obs['contact_base_policy']=self.obs['policy'][:,:195].clone();self.last=None
 def get_observations(self):return self.obs
 def reset(self):self.i=0;self.action_filter.reset(torch.ones(4,device=self.device,dtype=torch.bool));return self.obs,{}
 def step(self,a):
  self.last=self.action_filter(a).clone();self.i+=1
  dones=torch.tensor([self.i%3==0,self.i%5==0,self.i%7==0,self.i%11==0],device=self.device)
  self.action_filter.reset(dones)
  self.obs={k:v+.0001 for k,v in self.obs.items()}
  return self.obs,torch.zeros(4,device=self.device),dones,{}
for device in ('cpu','cuda'):
 if device=='cuda' and not torch.cuda.is_available():continue
 torch.manual_seed(20261026)
 env=Fake(device);teacher=torch.jit.trace(Teacher().to(device),torch.zeros(4,195,device=device));command=types.SimpleNamespace(cfg=types.SimpleNamespace(body_names=['base_link']),body_lin_vel_w=torch.zeros(4,1,3,device=device))
 wrapped=TrainingSpeedBlend(env,teacher,command,{},64)
 try:wrapped.step(torch.zeros(4,30,device=device));raise AssertionError('Missing obs accepted')
 except RuntimeError:pass
 deployed=ReferenceSpeedBlend.__new__(ReferenceSpeedBlend);deployed.command=command;deployed.index=0;deployed.maximum=1.;deployed.low=.05;deployed.high=.1;deployed.hip_filter=None;deployed.leg_filter=None;deployed.trace=[];deployed.audit={};deployed.base=types.SimpleNamespace(teacher=teacher)
 outer=VectorAnkleFilter(names,4,device,.5)
 obs=wrapped.get_observations();saved_rng=torch.get_rng_state().clone();saved_gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 for step in range(256):
  speed=torch.tensor([0.,.04,.075,.12],device=device).roll(step%4);command.body_lin_vel_w[:,0,0]=speed
  raw=(torch.arange(120,device=device,dtype=torch.float32).reshape(4,30)+step)/100
  original=raw.clone();before={k:v.clone() for k,v in obs.items()};want=deployed(obs,raw);want=outer(want)
  original_tf32=torch.backends.cuda.matmul.allow_tf32
  obs,_,dones,_=wrapped.step(raw)
  assert torch.backends.cuda.matmul.allow_tf32==original_tf32
  torch.testing.assert_close(env.last,want,rtol=0,atol=0);outer.reset(dones)
  assert torch.equal(raw,original)
  if step%23==0:wrapped.get_observations()
 assert torch.equal(saved_rng,torch.get_rng_state())
 if device=='cuda':assert torch.equal(saved_gpu,torch.cuda.get_rng_state())
 assert len(wrapped.trace)==64 and wrapped.completed_steps==256 and wrapped.zero_samples>0 and wrapped.active_samples>0
 wrapped.episode_length_buf=torch.ones(4,device=device);assert torch.equal(env.episode_length_buf,torch.ones(4,device=device))
 wrapped.reset();assert wrapped.current is not None
 print(device,'256-step exact deployed parity with asynchronous resets and unchanged RNG/actions')
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);export=p/'teacher.pt';cp=p/'teacher_checkpoint.pt';torch.jit.trace(Teacher(),torch.zeros(1,195)).save(str(export));cp.write_bytes(b'checkpoint')
 c=make_contract(export,cp);j={'reference_speed_teacher_export':str(export),'maximum_candidate_weight':1.,'blend_speed_low':.05,'blend_speed_high':.1,'leg_filter_alpha':.5,'leg_filter_scope':'ankles'}
 assert validate_jobs(c,[j])['matched_batch_jobs']==1;validate_resume(c,copy.deepcopy(c))
 for key,value in [('maximum_candidate_weight',.5),('blend_speed_low',.04),('blend_release_tau',0.),('reference_speed_teacher_export',str(cp)),('leg_filter_alpha',.8)]:
  bad=dict(j);bad[key]=value
  try:validate_jobs(c,[bad]);raise AssertionError(key+' accepted')
  except ValueError:pass
 for jobs in (None,[]):
  try:validate_jobs(c,jobs);raise AssertionError('Implicit replay accepted')
  except ValueError:pass
 bad=copy.deepcopy(c);bad['maximum_candidate_weight']=.5
 try:validate_resume(c,bad);raise AssertionError('Changed resume accepted')
 except ValueError:pass
 assert validate_checkpoint_jobs(p/'legacy.pt',[]) is None
print('Contract mismatch rejection and legacy default verified')
