"""Frozen matched ten-model continuation queue; no adaptive budgets or checkpoint search."""
import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO)
 assert json.loads((ROOT/'actor_pretraining_validation.json').read_text())['passed']
 assert json.loads((ROOT/'preflight_validation.json').read_text())['passed']
 cohort=json.loads((ROOT/'cohort.json').read_text())
 frozen=json.loads((ROOT/'frozen_training_sources.json').read_text())
 assert json.loads((ROOT/'infrastructure_interruption/retry_authorization.json').read_text())['authorized']
 while not (ROOT/'nine_models_training_complete').exists():
  assert not (ROOT/'stop').exists();time.sleep(20)
 done=json.loads((ROOT/'training_records.json').read_text());assert len(done)==9
 assert not any(r['group']=='sym_cont' and r['training_seed']==1 for r in done)
 for job in cohort['jobs']:
  if not (job['group']=='sym_cont' and job['training_seed']==1):continue
  assert not (ROOT/'stop').exists()
  for p,h in frozen['files'].items():assert sha(p)==h,p
  parent=cohort['parents'][str(job['training_seed'])]
  cp=Path(parent['checkpoint']);assert sha(cp)==parent['sha256']
  assert shutil.disk_usage(ROOT).free>350*1024*1024,'Preserve evidence; disk capacity low'
  name=job['group']+'_s'+str(job['training_seed']);out=ROOT/'training'/name
  out.mkdir(parents=True,exist_ok=True)
  args=['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','1500','--run_name','issue19_20261113_'+name+'_authorized_retry','--seed',str(job['training_seed']),'--resume','--load_run',cp.parent.name,'--checkpoint',cp.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation','requested','--post_push_pitch_weight','0.0','--issue12_audit_dir',str(out),'--issue17_mode','sym_int','--issue19_mode',job['group'],'agent.save_interval=1500']
  started=time.time()
  with (out/'console.log').open('w') as log:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue19train',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'training_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),child_pid=child.pid,state='training',name=name,completed=len(done),total=10,started=started,args=args)))
   while child.poll() is None:
    if (ROOT/'stop').exists():
     os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Explicit stop requested')
    time.sleep(5)
  assert child.returncode==0,(name,child.returncode)
  summary=json.loads((out/'run_summary.json').read_text())
  assert summary['additional_updates']==1500 and summary['environment_steps']==36000 and summary['num_envs']==4096 and not summary['std_reset']
  loaded=json.loads((out/'loaded_state_audit.json').read_text());assert loaded['passed'] and loaded['checkpoint_sha256']==parent['sha256']
  final=Path(summary['log_dir'])/('model_'+str(summary['final_iteration'])+'.pt');assert final.is_file()
  done.append(dict(**job,checkpoint=str(final),sha256=sha(final),parent_sha256=parent['sha256'],wall_seconds=time.time()-started,summary=summary))
  (ROOT/'training_records.json').write_text(json.dumps(done,indent=2)+'\n')
  print('ISSUE19_FORMAL_TRAINING_DONE',name,len(done),'of10',flush=True)
 assert len(done)==10
 (ROOT/'formal_training_complete').write_text('Exactly10 models x1500updates x4096envs; final checkpoints only.\n')
 (ROOT/'training_progress.json').write_text(json.dumps(dict(state='complete',completed=10,total=10,driver_pid=os.getpid())))
if __name__=='__main__':main()
