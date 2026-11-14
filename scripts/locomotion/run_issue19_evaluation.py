"""Sixty fixed paired formal evaluations after ten issue19 continuations."""
import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO)
 while not (ROOT/'formal_training_complete').exists():
  assert not (ROOT/'stop').exists()
  (ROOT/'evaluation_progress.json').write_text(json.dumps(dict(state='waiting_for_training',driver_pid=os.getpid())))
  time.sleep(20)
 assert json.loads((ROOT/'evaluation_preflight_validation.json').read_text())['passed']
 records=json.loads((ROOT/'training_records.json').read_text());assert len(records)==10
 models={r['group']+'_s'+str(r['training_seed']):r for r in records}
 jobs=json.loads((ROOT/'evaluation_job_templates.json').read_text());assert len(jobs)==60
 frozen=json.loads((ROOT/'frozen_evaluation_sources.json').read_text())
 assert not (ROOT/'evaluation_records.json').exists(),'Audit existing partial results before resuming'
 done=[]
 for job in jobs:
  assert not (ROOT/'stop').exists()
  for p,h in frozen['files'].items():assert sha(p)==h,p
  model=models[job['model']];assert sha(model['checkpoint'])==model['sha256']
  assert shutil.disk_usage(ROOT).free>350*1024*1024,'Insufficient capacity: preserve all evidence'
  folder=ROOT/'evaluation'/job['protocol'];folder.mkdir(parents=True,exist_ok=True)
  out=folder/(job['name']+'.json');moments=folder/'moments'/job['name']
  args=['python','-u','scripts/locomotion/eval_issue19.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','1024','--seed',str(job['eval_seed']),'--checkpoint',model['checkpoint'],'--experiment_profile','robust08','--mass_randomization','bounded','--last_action_observation','requested','--issue5_condition',job['protocol'],'--extended_commands','--command_filter',','.join(job['commands']),'--paired_resets','--issue18_mode','original','--issue19_mode',job['group'],'--issue18_records',str(moments),'--output',str(out)]
  if job['protocol']=='combined':args+=['--with_pushes','--trial_risk_records','--failure_traces',str(folder/'traces'/job['name'])]
  started=time.time()
  with out.with_suffix('.log').open('w') as log:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue19eval',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'evaluation_progress.json').write_text(json.dumps(dict(state='evaluating',driver_pid=os.getpid(),child_pid=child.pid,name=job['name'],completed=len(done),total=60,started=started)))
   while child.poll() is None:
    if (ROOT/'stop').exists():os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Explicit stop requested')
    time.sleep(5)
  assert child.returncode==0,(job['name'],child.returncode)
  d=json.loads(out.read_text());assert d['issue19_actor']==job['group']
  assert d['issue17_clock']['integer_enabled'] and d['issue17_clock']['period_steps']==20
  assert d['issue18_intervention']['mode']=='original'
  assert d['issue18_tracking']['model_optimizer_state_unchanged']
  assert d['issue18_tracking']['issue18_actual_feedback_verified']
  assert d['issue18_tracking']['issue18_pre_action_phase_reconstruction_verified']
  assert [r['command'] for r in d['results']]==job['commands']
  assert all(r['num_envs']==1024 and r['seconds']==20 for r in d['results'])
  done.append(dict(**job,output=str(out),output_sha256=sha(out),checkpoint_sha256=model['sha256'],wall_seconds=time.time()-started))
  (ROOT/'evaluation_records.json').write_text(json.dumps(done,indent=2)+'\n')
  print('ISSUE19_FORMAL_EVAL_DONE',len(done),'of60',job['name'],flush=True)
 assert sum(r['initial_trials'] for r in done)==276480
 (ROOT/'formal_evaluation_complete').write_text('60 new jobs,270 command records,276480 initial trials.\n')
 (ROOT/'evaluation_progress.json').write_text(json.dumps(dict(state='complete',completed=60,total=60,driver_pid=os.getpid())))
if __name__=='__main__':main()
