"""Same-state deployed/training action parity, reset, noise and contracts."""
import sys,pathlib,types,copy,tempfile,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch,numpy as np
from training_speed_blend import TrainingSpeedBlend,validate_resume,validate_jobs,validate_checkpoint_jobs,make_contract
from reference_speed_blend import ReferenceSpeedBlend
from training_action_filter import VectorAnkleFilter
from candidate_action_scope import make_scope_contract
names=json.loads(pathlib.Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/videos/validation/manifest.json').read_text())['joint_names']
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
 scope=make_scope_contract(names,'lower_body_waist');contract={'version':2,'candidate_action_scope':scope};frozen=torch.tensor([n in scope['teacher_joint_names'] for n in names],device=device)
 wrapped=TrainingSpeedBlend(env,teacher,command,contract,64,action_names=names)
 try:wrapped.step(torch.zeros(4,30,device=device));raise AssertionError('Missing obs accepted')
 except RuntimeError:pass
 deployed=ReferenceSpeedBlend.__new__(ReferenceSpeedBlend);deployed.command=command;deployed.index=0;deployed.maximum=1.;deployed.low=.05;deployed.high=.1;deployed.hip_filter=None;deployed.leg_filter=None;deployed.trace=[];deployed.audit={};deployed.base=types.SimpleNamespace(teacher=teacher);deployed.action_scope=scope
 outer=VectorAnkleFilter(names,4,device,.5)
 obs=wrapped.get_observations();saved_rng=torch.get_rng_state().clone();saved_gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 for step in range(256):
  speed=torch.tensor([0.,.04,.075,.12],device=device).roll(step%4);command.body_lin_vel_w[:,0,0]=speed
  raw=(torch.arange(120,device=device,dtype=torch.float32).reshape(4,30)+step)/100
  original=raw.clone();before={k:v.clone() for k,v in obs.items()};want=deployed(obs,raw);want=outer(want)
  original_tf32=torch.backends.cuda.matmul.allow_tf32
  obs,_,dones,_=wrapped.step(raw)
  assert torch.backends.cuda.matmul.allow_tf32==original_tf32
  torch.testing.assert_close(env.last,want,rtol=0,atol=0);assert torch.equal(env.last[:,frozen],teacher(before['contact_base_policy'])[:,frozen]);outer.reset(dones)
  assert torch.equal(raw,original)
  if step%23==0:wrapped.get_observations()
 assert torch.equal(saved_rng,torch.get_rng_state())
 if device=='cuda':assert torch.equal(saved_gpu,torch.cuda.get_rng_state())
 assert len(wrapped.trace)==64 and wrapped.completed_steps==256 and wrapped.zero_samples>0 and wrapped.active_samples>0
 wrapped.episode_length_buf=torch.ones(4,device=device);assert torch.equal(env.episode_length_buf,torch.ones(4,device=device))
 wrapped.reset();assert wrapped.current is not None
 print(device,'256-step scoped training/deployed parity;13 exact teacher outputs, asynchronous resets and unchanged RNG/actions')

with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);export=p/'teacher.pt';cp=p/'teacher_checkpoint.pt';torch.jit.trace(Teacher(),torch.zeros(1,195)).save(str(export));cp.write_bytes(b'checkpoint')
 legacy=make_contract(export,cp);scoped=make_contract(export,cp,names,'lower_body_waist')
 job={'reference_speed_teacher_export':str(export),'maximum_candidate_weight':1.,'blend_speed_low':.05,'blend_speed_high':.1,'leg_filter_alpha':.5,'leg_filter_scope':'ankles','candidate_action_scope':'lower_body_waist'}
 result=validate_jobs(scoped,[job]);assert result['contract']==scoped and scoped['version']==2
 validate_resume(scoped,copy.deepcopy(scoped))
 for c,j in [(legacy,job),(scoped,{k:v for k,v in job.items() if k!='candidate_action_scope'}),(scoped,dict(job,candidate_action_scope='all'))]:
  try:validate_jobs(c,[j])
  except ValueError:pass
  else:raise AssertionError('Checkpoint/replay scope mismatch accepted')
 try:validate_resume(scoped,legacy)
 except ValueError:pass
 else:raise AssertionError('Silent full-action resume accepted')
print('Integrated version2/base1 validation, original teacher binding fields and resume/replay mismatch guards passed')
