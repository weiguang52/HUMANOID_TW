"""Conditional stepB: own-checkpoint full-state continuation, fixed five seeds."""
import json,hashlib,os
from pathlib import Path
from run_issue10 import ROOT,REPO,run,evaluate

def main():
 os.chdir(REPO);g=json.loads((ROOT/'gate_decision.json').read_text());assert (ROOT/'stepA_evaluation_complete').exists()
 if not g['gate_passed']:
  (ROOT/'stepB_skipped').write_text('Frozen gate failed: no training launched.');print('STEP_B_SKIPPED',flush=True);return
 models=json.loads((ROOT/'input_models.json').read_text());statepath=ROOT/'continuation_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
 for seed in range(1,6):
  name=f'requested_s{seed}';parent=models[f'{name}_u1500'];p=Path(parent['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==parent['sha256']
  if name not in state:
   runname='issue10_20261109_'+name
   if list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+runname)):raise RuntimeError('Incomplete continuation requires explicit recovery')
   args=['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','3000','--run_name',runname,'--seed',str(seed),'--resume','--load_run',p.parent.name,'--checkpoint',p.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation','requested']
   run(args,ROOT/f'{name}_continue_train.log');dirs=list((REPO/'logs/rsl_rl/tw_stage1').glob('*_'+runname));assert len(dirs)==1
   state[name]={'seed':seed,'run_dir':str(dirs[0]),'parent_checkpoint':str(p),'parent_sha256':parent['sha256'],'additional_updates':3000,'full_optimizer_resume':True,'std_reset':False};statepath.write_text(json.dumps(state,indent=2))
  folder=Path(state[name]['run_dir'])
  for target,idx,actual in [(3000,10500,3006),(4500,11994,4500)]:
   cp=folder/f'model_{idx}.pt';assert cp.exists();mn=f'{name}_u{target}';models[mn]={'training_seed':seed,'target_updates':target,'actual_logged_updates':actual,'iteration':idx,'checkpoint':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()};(ROOT/'input_models.json').write_text(json.dumps(models,indent=2))
   for es in [1,2,3]:
    out=ROOT/f'{mn}_eval{es}.json'
    if not out.exists():run(evaluate(cp,es,out),out.with_suffix('.log'))
 (ROOT/'stepB_evaluation_complete').write_text('complete');run(['python','scripts/locomotion/analyze_issue10.py'],ROOT/'stepB_analysis.log');print('ISSUE10_STEP_B_COMPLETE',flush=True)
if __name__=='__main__':main()
