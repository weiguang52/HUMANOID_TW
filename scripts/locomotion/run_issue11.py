"""Issue11 no-training evaluation pipeline; fixed cohort and protocol."""
import json,hashlib,subprocess,time,os,csv
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue11_20261110');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')

def run(args,log):
 if (ROOT/'stop').exists():raise RuntimeError('stop requested')
 assert 'train.py' not in ' '.join(args)
 print('START',log.name,flush=True)
 with log.open('w') as out:
  p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue11',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
  progress={'pid':p.pid,'log':str(log),'args':args,'state':'running','started_unix':time.time()};(ROOT/'progress.json').write_text(json.dumps(progress,indent=2));p.wait()
 progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time());(ROOT/'progress.json').write_text(json.dumps(progress,indent=2));print('END',log.name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(log)

def evaluate(cp,seed,out,num_envs=1024):
 return ['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs',str(num_envs),'--seed',str(seed),'--checkpoint',str(cp),'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--mass_randomization','bounded','--last_action_observation','requested','--extended_commands','--command_filter','forward_03,forward_05','--paired_resets','--output',str(out)]

def main():
 os.chdir(REPO);(ROOT/'runner_process.json').write_text(json.dumps({'pid':os.getpid()}));models=json.loads((ROOT/'input_models.json').read_text())
 for e in models.values():assert hashlib.sha256(Path(e['checkpoint']).read_bytes()).hexdigest()==e['sha256']
 (ROOT/'smoke_complete_rng').mkdir(exist_ok=True)
 for kind in ['baseline','expanded']:
  out=ROOT/'smoke_complete_rng'/f'{kind}.json'
  if not out.exists():
   args=evaluate(models['requested_s1_u4500']['checkpoint'],1,out,64)+['--failure_traces',str(ROOT/'smoke_complete_rng'/f'{kind}_traces'),'--observation_audit',str(ROOT/'smoke_complete_rng'/f'{kind}_state')]
   if kind=='expanded':args+=['--trial_risk_records']
   run(args,ROOT/f'smoke_{kind}.log')
 run(['python','scripts/locomotion/audit_issue11_smoke.py'],ROOT/'smoke_audit.log')
 assert json.loads((ROOT/'observer_equivalence_audit.json').read_text())['passed']
 baseline=list(csv.DictReader((REPO/'validation_artifacts/stage1_issue10_20261109/all_evaluation_rates.csv').open()));comparisons={}
 for stage in [4500,1500]:
  for seed in range(1,6):
   name=f'requested_s{seed}_u{stage}'
   for es in [1,2,3]:
    out=ROOT/f'{name}_eval{es}.json'
    if not out.exists():run(evaluate(models[name]['checkpoint'],es,out)+['--failure_traces',str(ROOT/'traces'/out.stem),'--trial_risk_records'],out.with_suffix('.log'))
    d=json.loads(out.read_text());assert d['paired_resets'] and d['deterministic_policy'] and d['mass_randomization']=='bounded' and d['last_action_observation']=='requested'
    assert d['policy_last_action_params']=={} and d['critic_last_action_params']=={}
    cc=[]
    for x in d['results']:
     assert x['num_envs']==1024 and x['seconds']==20 and x['seed']==es
     old=next(r for r in baseline if int(r['training_seed'])==seed and int(r['target_updates'])==stage and int(r['eval_seed'])==es and r['command']==x['command']);cc.append({'command':x['command'],'old_falls':int(old['falls']),'new_falls':x['failed_trials'],'equal':int(old['falls'])==x['failed_trials']})
    comparisons[out.name]=cc;(ROOT/'baseline_count_comparison.json').write_text(json.dumps(comparisons,indent=2))
 (ROOT/'all_evaluation_complete').write_text('complete');print('ISSUE11_ALL_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
