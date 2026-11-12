"""Issue16 preflight: actual original evaluator parity and reflection/default-target sanity."""
import json,os
import numpy as np
from issue16_common import ROOT,REPO,run,evaluate_args,sha

def main():
 os.chdir(REPO);cohort=json.loads((ROOT/'cohort.json').read_text());model=cohort['models']['control6000_s1']
 folder=ROOT/'smoke';folder.mkdir(exist_ok=True)
 checks={}
 for mode in ['plain','identity','mirror','default']:
  output=folder/(mode+'.json')
  commands=['forward_03','stand'] if mode=='mirror' else ['stand'] if mode=='default' else ['forward_03']
  args=evaluate_args(model,1,output,num_envs=64,commands=commands,plain=mode=='plain',observer=mode!='plain',mode=mode if mode!='plain' else 'original')
  args+=['--observation_audit',str(folder/(mode+'_state'))]
  if not output.exists():run(args,output.with_suffix('.log'),'smoke_'+mode)
  data=json.loads(output.read_text())
  assert data['observation_dims']=={'policy':[101],'critic':[134]}
  if mode!='plain':
   assert data['issue16_tracking']['model_optimizer_state_unchanged']
   assert data['issue16_intervention']['mode']==mode
   assert data['issue16_intervention']['calls']==len(commands)*1000
   if mode=='default':assert data['issue16_tracking']['default_target_error_max']==0
  checks[mode]=dict(passed=True,commands=len(commands),checkpoint_sha256=sha(model['checkpoint']),formal_trials=0,training_updates=0)
 a=json.loads((folder/'plain.json').read_text());b=json.loads((folder/'identity.json').read_text())
 assert a['results']==b['results']
 x=dict(np.load(folder/'plain_state/forward_03.npz'));y=dict(np.load(folder/'identity_state/forward_03.npz'))
 assert x.keys()==y.keys()
 for name in x:assert x[name].dtype==y[name].dtype and x[name].shape==y[name].shape and x[name].tobytes()==y[name].tobytes(),name
 (ROOT/'smoke_audit.json').write_text(json.dumps(dict(passed=True,checks=checks,original_metrics_exact=True,final_root_joint_state_and_all_rng_bitwise_identical=True,compared_state_fields=list(x),formal_trials=0,training_updates=0),indent=2)+'\n')
 print('ISSUE16_SMOKE_PASS',flush=True)
if __name__=='__main__':main()
