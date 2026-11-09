"""Audit continuation recipe and source checkpoint/optimizer continuity."""
import json,re,hashlib,inspect
from pathlib import Path
import yaml,torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')

def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:yield path+'/'+k,a.get(k),b.get(k)
   else:yield from diff(a[k],b[k],path+'/'+k)
 elif a!=b:yield path,a,b

def audit():
 g=json.loads((ROOT/'gate_decision.json').read_text());records={};sources=json.loads((ROOT/'source_runs.json').read_text())
 state=json.loads((ROOT/'continuation_state.json').read_text()) if (ROOT/'continuation_state.json').exists() else {}
 if g['gate_passed']:assert len(state)==5 and (ROOT/'stepB_evaluation_complete').exists()
 else:assert not state and not list(Path('logs/rsl_rl/tw_stage1').glob('*_issue10_20261109_requested_s*'))
 for name,e in state.items():
  old=Path(sources[str(e['seed'])]['run_dir']);new=Path(e['run_dir']);record={}
  for kind in ['env','agent']:
   a=yaml.load((old/f'params/{kind}.yaml').read_text(),Loader=yaml.BaseLoader);b=yaml.load((new/f'params/{kind}.yaml').read_text(),Loader=yaml.BaseLoader)
   assert b['seed']==str(e['seed']);changes=list(diff(a,b));allowed={'/log_dir'} if kind=='env' else {'/run_name','/max_iterations','/load_run','/load_checkpoint'}
   assert all(x[0] in allowed for x in changes),(name,kind,changes)
   if kind=='agent':assert b['max_iterations']=='3000' and b['load_run']==old.name and b['load_checkpoint']=='model_8995.pt' and b['resume']=='true'
   record[kind]=changes
  text=(ROOT/f'{name}_continue_train.log').read_text();iters=re.findall(r'Learning iteration (\d+)/(\d+)',text);assert len(iters)==3000 and iters[0][0]=='8995' and iters[-1][0]=='11994'
  assert '[FT] Reset' not in text
  parent=torch.load(e['parent_checkpoint'],map_location='cpu',weights_only=False);final=torch.load(new/'model_11994.pt',map_location='cpu',weights_only=False)
  pst=parent['optimizer_state_dict']['state'];fst=final['optimizer_state_dict']['state'];assert set(pst)==set(fst)
  increments=[float(fst[k]['step'])-float(pst[k]['step']) for k in pst if 'step' in pst[k]];assert increments and min(increments)>0
  record['logged_updates']=len(iters);record['optimizer_step_increase_min']=min(increments);record['optimizer_step_increase_max']=max(increments);records[name]=record
 semantics={'runner_load_source':inspect.getsource(OnPolicyRunner.load),'ppo_load_source':inspect.getsource(PPO.load),'train_call':'runner.load(resume_path), default load_cfg=None; remote_env clears TW_RESET_STD'}
 (ROOT/'config_continuation_audit.json').write_text(json.dumps({'passed':True,'gate_passed':g['gate_passed'],'continuations':records,'resume_semantics':semantics},indent=2));print('CONFIG_CONTINUATION_PASS',len(records))
if __name__=='__main__':audit()
