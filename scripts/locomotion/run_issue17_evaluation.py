"""Fixed original/mirrored/combined evaluation queue for the ten final models."""
import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO);assert (ROOT/'formal_training_complete').exists()
 records=json.loads((ROOT/'training_records.json').read_text());assert len(records)==10
 models={f"{r['group']}_s{r['training_seed']}":r for r in records}
 jobs=json.loads((ROOT/'evaluation_job_templates.json').read_text());assert len(jobs)==90
 assert not (ROOT/'evaluation_records.json').exists(),'Audit existing partial queue before resuming'
 frozen=json.loads((ROOT/'frozen_training_sources.json').read_text());done=[]
 for job in jobs:
  assert not (ROOT/'stop').exists()
  for p,h in frozen['files'].items():assert sha(p)==h,p
  m=models[job['model']];assert sha(m['checkpoint'])==m['sha256']
  assert shutil.disk_usage(ROOT).free>300*1024*1024,'Insufficient disk capacity: preserve existing evidence'
  folder=ROOT/'evaluation'/job['protocol'];folder.mkdir(parents=True,exist_ok=True);out=folder/(job['name']+'.json');moments=folder/'moments'/job['name']
  args=['python','-u','scripts/locomotion/eval_issue17.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(job['eval_seed']),'--checkpoint',m['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition','combined' if job['protocol']=='combined' else 'clean','--extended_commands','--command_filter',','.join(job['commands']),'--paired_resets','--issue16_mode',job['mode'],'--issue16_records',str(moments),'--output',str(out)]
  if job['protocol']=='combined':args+=['--with_pushes','--trial_risk_records','--failure_traces',str(folder/'traces'/job['name'])]
  start=time.time()
  with out.with_suffix('.log').open('w') as log:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17eval',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'evaluation_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),child_pid=child.pid,name=job['name'],completed=len(done),started_unix=start,log=str(out.with_suffix('.log')),args=args)))
   while child.poll() is None:
    if (ROOT/'stop').exists():os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Explicit stop requested')
    time.sleep(5)
  assert child.returncode==0,(job['name'],child.returncode)
  d=json.loads(out.read_text());assert d['issue17_clock']['integer_enabled'] and d['issue17_clock']['period_steps']==20
  assert d['issue16_intervention']['mode']==job['mode'] and d['issue16_tracking']['model_optimizer_state_unchanged']
  assert {r['command'] for r in d['results']}==set(job['commands'])
  assert all(r['num_envs']==1024 and r['seconds']==20 for r in d['results'])
  done.append(dict(**job,output=str(out),output_sha256=sha(out),checkpoint_sha256=m['sha256'],wall_seconds=time.time()-start))
  (ROOT/'evaluation_records.json').write_text(json.dumps(done,indent=2)+'\n');print('FORMAL_EVAL_COMPLETE',len(done),'of90',job['name'],flush=True)
 assert sum(r['initial_trials'] for r in done)==460800
 (ROOT/'formal_evaluation_complete').write_text('90jobs450commandrecords460800initialtrials complete; statistical/manual acceptance audit remains.\n')
if __name__=='__main__':main()
