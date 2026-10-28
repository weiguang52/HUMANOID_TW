"""Actual RSL storage/update and projected Gaussian invariants."""
import copy, inspect, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'rsl_rl'))
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from anchored_ppo import AnchoredPPO, install_anchor
from controlled_action_likelihood import ACTIVE,make_contract,selected_indices,projected_statistics,validate_resume
from controlled_likelihood_ppo import ControlledLikelihoodPPO,install_controlled_likelihood
from candidate_action_scope import EXPECTED_NAMES

def rejects(call):
 try:call()
 except (ValueError,RuntimeError):return
 raise AssertionError('Invalid input accepted')

names=list(EXPECTED_NAMES);contract=make_contract(names);idx=selected_indices(contract)
frozen=[i for i in range(30) if i not in idx]
validate_resume(None,contract,allow_teacher_initialization=True)
validate_resume(copy.deepcopy(contract),contract)
rejects(lambda:validate_resume(None,contract))
rejects(lambda:validate_resume(dict(contract,KL='changed'),contract))
base=inspect.getsource(AnchoredPPO.update);new=inspect.getsource(ControlledLikelihoodPPO.update)
restored=new.replace('actions_log_prob, controlled_entropy = self.controlled_statistics(batch.observations, batch.actions)','actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore').replace('entropy = controlled_entropy[:original_batch_size]','entropy = self.actor.output_entropy[:original_batch_size]')
assert restored==base

for device in ['cpu','cuda']:
 if device=='cuda' and not torch.cuda.is_available():continue
 torch.manual_seed(5617)
 mean=torch.randn(12,30,device=device,requires_grad=True)
 std=torch.full_like(mean,.7,requires_grad=True);actions=torch.randn_like(mean).detach()
 active=(torch.arange(12,device=device)%3!=0).float().unsqueeze(-1)
 rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 lp,ent=projected_statistics(mean,std,actions,active,idx)
 expected=torch.distributions.Normal(mean[:,idx],std[:,idx])
 torch.testing.assert_close(lp,torch.where(active[:,0].bool(),expected.log_prob(actions[:,idx]).sum(-1),0.),rtol=0,atol=0)
 torch.testing.assert_close(ent,torch.where(active[:,0].bool(),expected.entropy().sum(-1),0.),rtol=0,atol=0)
 cm=mean.detach().clone();cm[:,frozen]+=2
 cs=std.detach().clone();cs[:,frozen]*=2
 ca=actions.clone();ca[:,frozen]+=4
 changed=projected_statistics(cm,cs,ca,active,idx)
 assert torch.equal(lp,changed[0]) and torch.equal(ent,changed[1])
 gm,gs=torch.autograd.grad((lp+.01*ent).mean(),(mean,std))
 assert torch.count_nonzero(gm[:,frozen])==torch.count_nonzero(gs[:,frozen])==0
 assert torch.count_nonzero(gm[~active[:,0].bool()])==torch.count_nonzero(gs[~active[:,0].bool()])==0
 assert torch.count_nonzero(gm[active[:,0].bool()][:,idx])>0
 offlp,offent=projected_statistics(mean,std,actions,torch.zeros_like(active),idx)
 assert torch.count_nonzero(offlp)==torch.count_nonzero(offent)==0
 assert torch.equal(rng,torch.random.get_rng_state())
 if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
 permutation=list(reversed(range(30)));pc=make_contract([names[i] for i in permutation])
 perm=projected_statistics(mean[:,permutation],std[:,permutation],actions[:,permutation],active,selected_indices(pc))
 torch.testing.assert_close(lp,perm[0],atol=4e-6,rtol=1e-6)
 rejects(lambda:projected_statistics(mean,std,actions,active+.1,idx))
 rejects(lambda:projected_statistics(mean,std,actions,active,idx[:-1]))
 rejects(lambda:projected_statistics(mean,std*0,actions,active,idx))
 print(device,'projected formula, ignored/inactive invariance, gradients, ordering and RNG passed')

 obs=TensorDict({'policy':torch.randn(16,203,device=device),'critic':torch.randn(16,195,device=device),ACTIVE:(torch.arange(16,device=device)%2==0).float().unsqueeze(-1)},batch_size=[16])
 groups={'actor':['policy'],'critic':['critic']}
 actor=MLPModel(obs,groups,'actor',30,hidden_dims=[16],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5}).to(device)
 critic=MLPModel(obs,groups,'critic',1,hidden_dims=[16]).to(device)
 storage=RolloutStorage('rl',16,4,obs,[30],device)
 alg=PPO(actor,critic,storage,num_learning_epochs=2,num_mini_batches=2,schedule='adaptive',device=device)
 install_anchor(alg,1.);install_controlled_likelihood(alg,contract)
 teacher={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()}
 for step in range(4):
  obs[ACTIVE]=((torch.arange(16,device=device)+step)%3!=0).float().unsqueeze(-1);before=obs[ACTIVE].clone()
  with torch.inference_mode():
   rng=torch.random.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
   reference=alg.actor(obs,stochastic_output=True).clone()
   next_rng=torch.random.get_rng_state().clone();next_gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
   torch.random.set_rng_state(rng)
   if gpu_rng is not None:torch.cuda.set_rng_state(gpu_rng)
   actual=alg.act(obs);assert torch.equal(reference,actual)
   assert torch.equal(next_rng,torch.random.get_rng_state())
   if next_gpu is not None:assert torch.equal(next_gpu,torch.cuda.get_rng_state())
   want,_=alg.controlled_statistics(obs,actual);assert torch.equal(alg.transition.actions_log_prob,want)
   oldlp=want.clone()
   alg.process_env_step(obs,torch.ones(16,device=device),torch.zeros(16,device=device,dtype=torch.bool),{})
   assert torch.equal(storage.observations[ACTIVE][step],before)
   assert torch.equal(storage.actions_log_prob[step,:,0],oldlp)
  obs=obs.clone();obs['policy']+=.01
 for batch in storage.mini_batch_generator(2,1):
  with torch.no_grad():
   alg.actor(batch.observations,stochastic_output=True)
   got,_=alg.controlled_statistics(batch.observations,batch.actions)
   torch.testing.assert_close(got,batch.old_actions_log_prob[:,0],atol=2e-5,rtol=1e-6)
 with torch.inference_mode():alg.compute_returns(obs)
 loss=alg.update();assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
 assert storage.step==0 and all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in teacher.items())
 assert all(p.grad is None for p in alg.anchor_teacher.parameters())
 assert alg.actor.mlp[0].weight.shape[1]==203 and alg.critic.mlp[0].weight.shape[1]==195
 print(device,'actual raw30 sample/RNG, rollout storage, shuffled old-score pairing, optimizer and frozen teacher passed')
print('Only two update source substitutions; full30 KL/scheduler/critic/anchors/optimizer/reductions unchanged')

# Version3 truthfully records training score while playback recovers exact v2 physical transform.
from controlled_action_likelihood import attach_training_contract,detach_training_contract
from candidate_scope_contract import split_training_scope,validate_scope_jobs
from training_speed_blend import make_contract as make_training_contract
import tempfile
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d);export=p/'teacher.pt';cp=p/'teacher_checkpoint.pt'
 export.write_bytes(b'teacher');cp.write_bytes(b'checkpoint')
 original=make_training_contract(export,cp,names,'lower_body_waist')
 added=make_training_contract(export,cp,names,'lower_body_waist',contract)
 assert added==attach_training_contract(original,contract)
 physical,restored=detach_training_contract(added)
 assert physical==original and restored==contract
 assert split_training_scope(added)==split_training_scope(original)
 validate_scope_jobs(added,[{'candidate_action_scope':'lower_body_waist'}])
 rejects(lambda:validate_scope_jobs(added,[{'candidate_action_scope':'all'}]))
 rejects(lambda:detach_training_contract(dict(added,PPO_action='wrong')))
 rejects(lambda:attach_training_contract(original,make_contract(list(reversed(names)))))
print('Version3 score metadata, exact base2/1 physical transform and resume/playback guards passed')
