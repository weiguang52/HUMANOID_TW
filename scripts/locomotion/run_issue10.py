"""Issue10 stepA only. Frozen gate controls a separate continuation driver."""
import json,hashlib,subprocess,time,os,shutil
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue10_20261109')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
COMMANDS=['forward_03','forward_05','forward_08']

def run(args,log):
 if (ROOT/'stop').exists():raise RuntimeError('stop requested')
 print('START',log.name,flush=True)
 with log.open('w') as out:
  p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue10',*args],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
  progress={'pid':p.pid,'log':str(log),'args':args,'state':'running','started_unix':time.time()};(ROOT/'progress.json').write_text(json.dumps(progress,indent=2));p.wait()
 progress.update(state='complete' if p.returncode==0 else 'failed',returncode=p.returncode,finished_unix=time.time());(ROOT/'progress.json').write_text(json.dumps(progress,indent=2))
 print('END',log.name,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(log)

def evaluate(checkpoint,seed,output):
 return ['python','-u','scripts/locomotion/eval_cmd.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(seed),'--checkpoint',str(checkpoint),'--experiment_profile','robust08','--with_pushes','--issue5_condition','combined','--mass_randomization','bounded','--last_action_observation','requested','--extended_commands','--command_filter',','.join(COMMANDS),'--paired_resets','--output',str(output)]

def stepA():
 os.chdir(REPO);(ROOT/'runnerA_process.json').write_text(json.dumps({'pid':os.getpid(),'completed':False}));models=json.loads((ROOT/'input_models.json').read_text());reused={}
 for seed in range(1,6):
  ref=REPO/'validation_artifacts'/('stage1_issue7_20261109' if seed<3 else 'stage1_issue9_20261109');prefix=f'new_bounded_s{seed}' if seed<3 else f'requested_s{seed}';manifest=json.loads((ref/'sha256.json').read_text())
  for es in [1,2,3]:
   src=ref/f'{prefix}_eval{es}.json';h=hashlib.sha256(src.read_bytes()).hexdigest();assert h==manifest[src.name]
   dst=ROOT/f'requested_s{seed}_u1500_eval{es}.json';shutil.copy2(src,dst);reused[dst.name]={'source':str(src),'sha256':h,'checkpoint_sha256':models[f'requested_s{seed}_u1500']['sha256']}
 (ROOT/'reused_evaluation_provenance.json').write_text(json.dumps(reused,indent=2))
 for stage in [500,1000]:
  for seed in range(1,6):
   name=f'requested_s{seed}_u{stage}';entry=models[name];assert hashlib.sha256(Path(entry['checkpoint']).read_bytes()).hexdigest()==entry['sha256']
   for es in [1,2,3]:
    out=ROOT/f'{name}_eval{es}.json'
    if not out.exists():run(evaluate(entry['checkpoint'],es,out),out.with_suffix('.log'))
 (ROOT/'stepA_evaluation_complete').write_text('complete')
 run(['python','scripts/locomotion/analyze_issue10.py'],ROOT/'stepA_analysis.log')
 (ROOT/'runnerA_process.json').write_text(json.dumps({'pid':os.getpid(),'completed':True}))
 print('ISSUE10_STEP_A_COMPLETE',flush=True)
if __name__=='__main__':stepA()
