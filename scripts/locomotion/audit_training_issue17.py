"""Audit every final formal run, actual augmentation calls and matched configurations."""
import hashlib,json
from pathlib import Path
import yaml
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def flat(v,p=''):
 if isinstance(v,dict):return {a:b for k,x in v.items() for a,b in flat(x,p+'/'+str(k)).items()}
 if isinstance(v,list):return {a:b for k,x in enumerate(v) for a,b in flat(x,p+'/'+str(k)).items()}
 return {p:v}
def readcfg(folder,name):return yaml.load((Path(folder)/'params'/(name+'.yaml')).read_text(),Loader=yaml.BaseLoader)
def main():
 assert (ROOT/'formal_training_complete').exists();records=json.loads((ROOT/'training_records.json').read_text());assert len(records)==10 and sum(r['updates'] for r in records)==15000;by={(r['group'],r['training_seed']):r for r in records};cohort=json.loads((ROOT/'cohort.json').read_text());audits=[]
 for r in records:
  name=f"{r['group']}_s{r['training_seed']}";folder=ROOT/'training'/name;loaded=json.loads((folder/'loaded_state_audit.json').read_text());summary=json.loads((folder/'run_summary.json').read_text());aug=json.loads((folder/'augmentation_runtime_audit.json').read_text());assert loaded['passed'] and summary['additional_updates']==1500 and summary['environment_steps']==36000 and summary['num_envs']==4096 and not summary['std_reset'];assert sha(r['checkpoint'])==r['sha256'];assert loaded['checkpoint_sha256']==cohort['models'][r['parent']]['sha256']
  assert aug['integer_clock'] and aug['updates']==1500 and aug['mode']==r['group']
  if r['group']=='sym_int':
   a=aug['audit'];assert a['training_batch_calls']==30000 and a['original_batch_sizes']==[24576] and a['augmented_batch_sizes']==[49152]
  else:assert aug['audit'] is None
  env=readcfg(summary['log_dir'],'env');assert env['issue17_integer_clock']=='true'
  old=readcfg(cohort['models'][f"control6000_s{r['training_seed']}"]['run_dir'],'env');a=flat(old);b=flat(env);diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)};assert set(diff)<= {'/log_dir','/issue17_integer_clock'},diff
  old_agent=flat(readcfg(cohort['models'][f"control6000_s{r['training_seed']}"]['run_dir'],'agent'));new_agent=flat(readcfg(summary['log_dir'],'agent'));agent_diff={k:[old_agent.get(k),new_agent.get(k)] for k in old_agent.keys()|new_agent.keys() if old_agent.get(k)!=new_agent.get(k)}
  assert all(k=='/run_name' or k.startswith('/algorithm/symmetry_cfg') for k in agent_diff),agent_diff
  audits.append(dict(name=name,loaded_parent_exact=True,updates=1500,augmentation=aug,old_control_environment_differences=diff,final_checkpoint_sha256=r['sha256']))
 for seed in range(1,6):
  for kind in ['env','agent']:
   a=flat(readcfg(by['control_int',seed]['summary']['log_dir'],kind));b=flat(readcfg(by['sym_int',seed]['summary']['log_dir'],kind));diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
   assert all(k in ['/log_dir','/run_name'] or (kind=='agent' and k.startswith('/algorithm/symmetry_cfg')) for k in diff),diff
 (ROOT/'formal_training_audit.json').write_text(json.dumps(dict(passed=True,formal_runs=10,formal_updates=15000,matched_seeds=5,records=audits,source_sha256=sha(__file__)),indent=2)+'\n');print('FORMAL_TRAINING_AUDIT_PASS10RUNS15000UPDATES',flush=True)
if __name__=='__main__':main()
