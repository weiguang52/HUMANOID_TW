"""Capture parity with actual RSL rollout/optimizer on CPU and CUDA."""
from pathlib import Path
import sys,copy,tempfile,random,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from anchored_ppo import install_anchor
from controlled_likelihood_ppo import install_controlled_likelihood
from controlled_action_likelihood import make_contract,ACTIVE,projected_statistics
from candidate_action_scope import EXPECTED_NAMES
from second_order_anchor import PREVIOUS,OLDER,VALID
from training_minibatch_capture import install_capture


def equal(a,b):
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert type(a) is type(b) and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b


def rng(device):
 return (torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone() if device=='cuda' else None,random.getstate(),np.random.get_state())


def restore(r,device):
 torch.set_rng_state(r[0])
 if device=='cuda':torch.cuda.set_rng_state(r[1])
 random.setstate(r[2]);np.random.set_state(r[3])


def equal_rng(a,b):
 equal(a[0],b[0]);equal(a[1],b[1]);assert a[2]==b[2]
 assert a[3][0]==b[3][0] and np.array_equal(a[3][1],b[3][1]) and a[3][2:]==b[3][2:]

for device in ['cpu','cuda']:
 if device=='cuda' and not torch.cuda.is_available():continue
 torch.manual_seed(5927);random.seed(5927);np.random.seed(5927)
 policy=torch.randn(16,203,device=device)
 obs=TensorDict({'policy':policy,'critic':torch.randn(16,315,device=device),PREVIOUS:policy.clone(),OLDER:policy.clone(),VALID:torch.zeros(16,1,device=device),ACTIVE:(torch.arange(16,device=device)%2==0).float().unsqueeze(-1)},batch_size=[16])
 groups={'actor':['policy'],'critic':['critic']}
 actor=MLPModel(obs,groups,'actor',30,hidden_dims=[16],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5}).to(device)
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=[16]).to(device)
 base=PPO(actor,critic,RolloutStorage('rl',16,4,obs,[30],device),num_learning_epochs=2,num_mini_batches=2,device=device)
 install_anchor(base,1.,temporal_coefficient=1.,temporal_order=2,temporal_target='student_mean')
 install_controlled_likelihood(base,make_contract(list(EXPECTED_NAMES)))
 other=copy.deepcopy(base)
 with tempfile.TemporaryDirectory(prefix='twrl_capture_test_') as temporary:
  path=Path(temporary)/'capture';original=other.storage
  cap=install_capture(other,path,slots=(0,3),provenance={'test':'synthetic actual RSL pipeline; no robot rollout'})
  assert cap.original is original and other.storage is original
  for step in range(4):
   obs=obs.clone();obs['policy']+=.01;obs[PREVIOUS]=obs['policy']-.01;obs[OLDER]=obs['policy']-.02;obs[VALID].fill_(float(step>=2))
   before=rng(device)
   with torch.inference_mode():
    actions=base.act(obs).clone();base.process_env_step(obs,torch.ones(16,device=device),torch.zeros(16,device=device,dtype=torch.bool),{})
   after=rng(device);restore(before,device)
   with torch.inference_mode():
    actual=other.act(obs);other.process_env_step(obs,torch.ones(16,device=device),torch.zeros(16,device=device,dtype=torch.bool),{})
   equal(actions,actual);equal_rng(after,rng(device))
  with torch.inference_mode():base.compute_returns(obs);other.compute_returns(obs)
  equal(base.storage.advantages,other.storage.advantages)
  assert float(other.storage.advantages.std())>0 and int(torch.count_nonzero(other.storage.advantages))>0
  before=rng(device);loss1=base.update();after=rng(device);restore(before,device);loss2=other.update();equal_rng(after,rng(device))
  equal(loss1,loss2);equal(base.actor.state_dict(),other.actor.state_dict());equal(base.critic.state_dict(),other.critic.state_dict());equal(base.optimizer.state_dict(),other.optimizer.state_dict())
  report=cap.finish();assert report['complete'] and report['yielded_batches']==4
  assert original.step==0 and base.storage.step==0
  first=torch.load(path/'batch_0000.pt',map_location='cpu',weights_only=False)
  assert first['observation_batch_size']==[32] and first['observations']['policy'].shape==(32,203) and first['observations']['critic'].shape==(32,315)
  assert first['observations'][ACTIVE].dtype==torch.float32 and first['observations'][VALID].shape==(32,1)
  assert first['observations'][PREVIOUS].shape==first['observations'][OLDER].shape==(32,203)
  assert len(first['controlled_action_indices'])==17 and len(first['old_distribution_params'])==2
  old_mean,old_std=first['old_distribution_params']
  old_score,_=projected_statistics(old_mean,old_std,first['actions'],first['observations'][ACTIVE],first['controlled_action_indices'])
  torch.testing.assert_close(old_score,first['old_actions_log_prob'][:,0],atol=3e-5,rtol=1e-6)
  assert first['old_distribution_params'][0].shape==first['actions'].shape==(32,30)
  for record in report['records']:
   p=Path(record['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'];snapshot=torch.load(p,map_location='cpu',weights_only=False)
   assert snapshot['batch_ordinal']==record['ordinal']
   assert all(not v.requires_grad and v.device.type=='cpu' for v in snapshot['observations'].values())
  for record in report['records']:
   after_path=Path(record['after_path']);assert hashlib.sha256(after_path.read_bytes()).hexdigest()==record['after_sha256']
  last_after=torch.load(path/'batch_0003.after.pt',map_location='cpu',weights_only=False)
  equal(last_after['actor_gradients'],{k:v.grad.detach().cpu() if v.grad is not None else None for k,v in other.actor.named_parameters()})
  equal(last_after['critic_gradients'],{k:v.grad.detach().cpu() if v.grad is not None else None for k,v in other.critic.named_parameters()})
  assert not list(path.glob('*.partial'))
  try:install_capture(other,Path(temporary)/'double',(0,))
  except ValueError:pass
  else:raise AssertionError('Double install accepted')
  for slots in [(),(1,0),(0,0),(-1,),(True,)]:
   try:install_capture(base,Path(temporary)/('invalid'+str(slots)),slots)
   except ValueError:pass
   else:raise AssertionError('Invalid slots accepted')
  assert other.storage is original
  base_storage=base.storage
  try:install_capture(base,path,(0,))
  except FileExistsError:pass
  else:raise AssertionError('Existing capture directory overwritten')
  assert base.storage is base_storage
  incomplete=install_capture(base,Path(temporary)/'incomplete',(4,))
  try:incomplete.finish()
  except RuntimeError:pass
  else:raise AssertionError('Uncaptured ordinals accepted')
  assert not (Path(temporary)/'incomplete/manifest.json').exists()
 print(device,'actual rollout/update/actor/critic/Adam/loss bit parity and all RNG streams passed; complete histories and score pairing captured without overwrites')
print('Capture installed before rollout/compute_returns; nonzero advantages and original storage identity preserved; real flag remains opt-in')
