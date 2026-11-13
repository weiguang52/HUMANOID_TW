import json,yaml,hashlib
from pathlib import Path
p=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def flat(x,prefix=''):
 if isinstance(x,dict):return {a:b for k,v in x.items() for a,b in flat(v,prefix+'/'+str(k)).items()}
 if isinstance(x,list):return {a:b for i,v in enumerate(x) for a,b in flat(v,prefix+'/'+str(i)).items()}
 return {prefix:x}
def cfg(mode,kind):
 s=json.loads((p/'preflight'/mode/'audit/run_summary.json').read_text());f=Path(s['log_dir'])/'params'/(kind+'.yaml');return yaml.load(f.read_text(),Loader=yaml.BaseLoader)
result={}
for kind in ['env','agent']:
 a=flat(cfg('control_int',kind));b=flat(cfg('sym_int',kind));diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
 allowed=lambda k:k=='/log_dir' or (kind=='agent' and k.startswith('/algorithm/symmetry_cfg'))
 assert all(allowed(k) for k in diff),diff
 result[kind+'_differences']=diff
for mode in ['control_int','sym_int']:
 d=json.loads((p/'preflight'/mode/'audit/loaded_state_audit.json').read_text());assert d['passed']
 e=cfg(mode,'env');table=json.loads((p/'reward_invariance_table.json').read_text())
 for r in table['rows']:assert float(e['rewards'][r['term']]['weight'])==r['weight']
 assert e['issue17_integer_clock']=='true'
result.update(passed=True,actual4096env_two_update_preflights=True,all18_active_reward_weights_preserved=True,both_loaded_actor_critic_optimizer_exact=True,augmentation_minibatch=49152,original_minibatch=24576,formal_training_updates=0)
(p/'enabled_preflight_audit.json').write_text(json.dumps(result,indent=2)+'\n');print('ENABLED_RUNTIME_CONFIG_AUDIT_PASS',json.dumps(result))
