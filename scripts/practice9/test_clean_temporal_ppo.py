from pathlib import Path
from types import SimpleNamespace
import sys,copy,importlib.util,torch,json
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from anchored_ppo import install_anchor
from controlled_likelihood_ppo import install_controlled_likelihood
from controlled_action_likelihood import ACTIVE,make_contract,projected_statistics,selected_indices
from clean_temporal_policy import CLEAN,configure_clean_temporal
from second_order_anchor import PREVIOUS,OLDER,VALID,student_mean_curvature_mse
b=Path('/root/twrl_runs/clean_curvature_20261027');names=json.load(open('/root/twrl_runs/controlled_curvature_20261027/formal_configuration_audit.json'))['new_training_contract']['candidate_action_scope']['action_joint_names'];contract=make_contract(names);indices=selected_indices(contract);torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('preclean_controlled',b/'source_before_clean_temporal/controlled_likelihood_ppo.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def make(device,coef=1):
 obs=TensorDict({'policy':torch.randn(32,203,device=device),'critic':torch.randn(32,315,device=device),CLEAN:torch.randn(32,203,device=device),PREVIOUS:torch.randn(32,203,device=device),OLDER:torch.randn(32,203,device=device),VALID:torch.ones(32,1,device=device),ACTIVE:(torch.arange(32,device=device)%2==0).float().unsqueeze(-1)},batch_size=[32]);obs[VALID][:2]=0
 groups={'actor':['policy'],'critic':['critic']};actor=MLPModel(obs,groups,'actor',30,hidden_dims=[16],distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution','init_std':.5}).to(device);critic=MLPModel(obs,groups,'critic',1,hidden_dims=[16]).to(device)
 with torch.no_grad():
  actions=actor(obs,stochastic_output=True);mean,std=actor.output_distribution_params;lp,_=projected_statistics(mean,std,actions,obs[ACTIVE],indices);batch=SimpleNamespace(observations=obs,actions=actions,old_actions_log_prob=lp.unsqueeze(-1),values=critic(obs),advantages=torch.randn(32,1,device=device),returns=torch.randn(32,1,device=device),old_distribution_params=tuple(x.clone() for x in actor.output_distribution_params),masks=None,hidden_states=(None,None))
 class Storage:
  def __init__(self):self.observations=obs;self.batch=batch
  def mini_batch_generator(self,*args):yield copy.deepcopy(self.batch)
  def clear(self):pass
 alg=PPO(actor,critic,Storage(),num_learning_epochs=1,num_mini_batches=1,schedule='adaptive',learning_rate=.001,device=device);install_anchor(alg,coef,temporal_coefficient=16.,temporal_order=2,temporal_target='student_mean');install_controlled_likelihood(alg,contract)
 return alg,obs,batch

def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def rng(device):return(torch.random.get_rng_state().clone(),torch.cuda.get_rng_state().clone() if device=='cuda' else None)
def restore(r):
 torch.random.set_rng_state(r[0])
 if r[1] is not None:torch.cuda.set_rng_state(r[1])
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 torch.manual_seed(5981);alg,obs,batch=make(device);baseline=copy.deepcopy(alg);baseline.__class__=old.ControlledLikelihoodPPO;r=rng(device);want=baseline.update();after=rng(device);restore(r);got=alg.update();assert same(want,got) and same(rng(device),after)
 assert same(alg.actor.state_dict(),baseline.actor.state_dict()) and same(alg.critic.state_dict(),baseline.critic.state_dict()) and same(alg.optimizer.state_dict(),baseline.optimizer.state_dict())
 alg,obs,batch=make(device,0);audit=configure_clean_temporal(alg,{'policy_dim':203});assert audit['all_three_temporal_inputs_clean'];batch.advantages.zero_();alg.value_loss_coef=0.;alg.entropy_coef=0.;alg.desired_kl=None;other=copy.deepcopy(alg);other.storage.batch.observations['policy']+=.25
 def objective(a):
  o=a.storage.batch.observations;views=[]
  for k in [CLEAN,PREVIOUS,OLDER]:v=o.clone(False);v['policy']=o[k];views.append(a.actor(v))
  return student_mean_curvature_mse(*views,o[VALID])
 with torch.no_grad():before=float(objective(alg))
 # Evaluating objective cached a distribution; clear cache before deepcopy is no longer needed.
 teacher={k:v.clone() for k,v in alg.anchor_teacher.state_dict().items()};r=rng(device);result=alg.update();after_rng=rng(device);restore(r);changed=other.update();assert same(after_rng,rng(device));assert same(alg.actor.state_dict(),other.actor.state_dict()),'Noisy current policy leaked into clean-only objective'
 with torch.no_grad():after=float(objective(alg))
 assert abs(result['anchor_temporal_mse']-before)<1e-6 and after<before
 assert all(torch.equal(v,alg.anchor_teacher.state_dict()[k]) for k,v in teacher.items()) and all(p.grad is None for p in alg.anchor_teacher.parameters())
 assert not alg.actor.obs_normalization and alg.actor.obs_groups==['policy'] and alg.critic.obs_groups==['critic']
 print(device,'default pre-edit full PPO loss/weights/critic/Adam/RNG exact; all3 clean objective/noisy current invariance,loss descent,frozen teacher passed',before,after)
print('Clean auxiliary keys stay outside NN groups; no simulator quality claim')
