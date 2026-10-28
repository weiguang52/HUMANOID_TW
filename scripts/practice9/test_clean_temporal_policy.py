from pathlib import Path
import sys,json,yaml,re,copy,torch
from types import SimpleNamespace
from tensordict import TensorDict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from clean_temporal_policy import make_contract,reconstruct,CLEAN,CleanPolicyObservations,CleanTwoPoliciesTraining,validate_resume_source
from second_order_anchor import PREVIOUS,OLDER,VALID
b=Path('/root/twrl_runs/training_loss_diagnostic_20261027');run=Path((b/'repaired_seed42/run').read_text().strip());cfg=yaml.load((run/'params/env.yaml').read_text(),Loader=yaml.BaseLoader);log=(b/'repaired_seed42/train.log').read_text()
def plain(v):
 if v=='null':return None
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,list):return [plain(x) for x in v]
 return v
terms=[]
for group in ['policy','critic']:
 section=log.split("Active Observation Terms in Group: '"+group+"'",1)[1];dims=re.findall(r'\|\s*\d+\s*\|\s*(\w+)\s*\|\s*\((\d+),\)',section)[:13];rows=[]
 for n,dim in dims:
  c=cfg['observations'][group][n];noise=plain(c['noise'])
  if noise:
   for k in ['n_min','n_max']:noise[k]=float(noise[k])
  rows.append({'name':n,'dim':int(dim),'func':c['func'],'params':plain(c['params']),'noise':noise,'scale':plain(c['scale']),'clip':plain(c['clip']),'modifiers':plain(c['modifiers']),'history_length':int(c['history_length'])})
 terms.append(rows)
contract=make_contract(*terms);assert make_contract(*terms)==json.loads(json.dumps(contract))
rec=json.load(open(run/'training_minibatches/manifest.json'))['records'][0];data=torch.load(rec['path'],map_location='cpu',weights_only=False)
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 obs=TensorDict({k:v.to(device) for k,v in data['observations'].items()},batch_size=data['observation_batch_size']);before=obs.clone();torch.rand(1,device=device);state=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone() if device=='cuda' else None
 out=reconstruct(obs,contract);assert torch.equal(state,torch.random.get_rng_state()) and (cuda is None or torch.equal(cuda,torch.cuda.get_rng_state()))
 assert all(torch.equal(obs[k],before[k]) for k in obs.keys());assert out.data_ptr()!=obs['policy'].data_ptr() and not torch.equal(out,obs['policy'])
 mask=torch.ones(203,dtype=torch.bool,device=device)
 for r in contract['replacements']:
  a,z=r['policy_range'];i,j=r['critic_range'];mask[a:z]=False;assert torch.equal(out[:,a:z],obs['critic'][:,i:j])
 assert torch.equal(out[:,mask],obs['policy'][:,mask])
 class Env:
  num_envs=4
  def __init__(self):self.device=device;self.i=0;self.episode_length_buf=torch.zeros(4,device=device);self.calls=0;self.pack()
  def pack(self):
   p=torch.full((4,203),float(self.i),device=device);c=torch.full((4,315),float(self.i)+.25,device=device);self.obs=TensorDict({'policy':p,'critic':c,'sentinel':p[:,:1].clone()},batch_size=[4])
  def get_observations(self):self.calls+=1;return self.obs
  def reset(self):self.i=0;self.pack();return self.obs,{}
  def step(self,a):self.i+=1;self.pack();done=torch.tensor([self.i%7==0,self.i%11==0,self.i%13==0,False],device=device);return self.obs,torch.ones(4,device=device),done,{}
 env=Env();wrapped=CleanTwoPoliciesTraining(CleanPolicyObservations(env,contract));o,_=wrapped.reset();history=[[o[CLEAN][i].clone()] for i in range(4)];count=[0]*4
 for step in range(64):
  saved=o.clone();calls=env.calls;again=wrapped.get_observations();assert env.calls==calls+1 and all(torch.equal(saved[k],again[k]) for k in saved.keys())
  actions=torch.ones(4,30,device=device);old_actions=actions.clone();o,r,done,_=wrapped.step(actions);assert torch.equal(actions,old_actions) and torch.equal(o['policy'],env.obs['policy']) and torch.equal(o['critic'],env.obs['critic']);assert not torch.equal(o[CLEAN],o['policy'])
  for i in range(4):
   if done[i]:count[i]=0;history[i]=[]
   else:count[i]=min(count[i]+1,2)
   assert bool(o[VALID][i])==(count[i]>=2)
   if count[i]>=2:assert torch.equal(o[PREVIOUS][i],history[i][-1]) and torch.equal(o[OLDER][i],history[i][-2])
   history[i].append(o[CLEAN][i].clone())
 print(device,'real recorded clean reconstruction/input+RNG,64-step repeated-get/async reset/noisy PPO retention passed')
for saved,requested in [({'temporal_coefficient':16,'temporal_observation_source':'clean_critic'},'policy'),({'temporal_coefficient':1},'clean_critic')]:
 try:validate_resume_source(saved,requested)
 except ValueError:pass
 else:raise AssertionError('silent observation source switch accepted')
validate_resume_source({},'clean_critic');validate_resume_source({'temporal_coefficient':16,'temporal_observation_source':'clean_critic'},'clean_critic')
for mutate in ['func','noise','dim','history']:
 p,c=copy.deepcopy(terms)
 if mutate=='func':c[2]['func']='wrong'
 elif mutate=='noise':p[0]['noise']=p[2]['noise']
 elif mutate=='dim':p[0]['dim']+=1
 else:p[2]['history_length']=1
 try:make_contract(p,c)
 except ValueError:pass
 else:raise AssertionError(mutate)
print('Contract roundtrip, invalid semantic/layout/noise/history and resume source guards passed; no training integration yet')
