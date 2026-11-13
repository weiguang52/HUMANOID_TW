"""Fixed ten-run integer-clock study; no checkpoint search or adaptive budget."""
import hashlib,json,os,shutil,subprocess,time,signal
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 os.chdir(REPO);cohort=json.loads((ROOT/'cohort.json').read_text())
 for n in ['disabled_two_update_reproduction.json','enabled_preflight_audit.json']:assert json.loads((ROOT/n).read_text())['passed']
 for n,count in [('transform_unit_tests.log',4),('integer_clock_tests.log',3)]:
  t=(ROOT/n).read_text();assert f'Ran {count} tests' in t and t.rstrip().endswith('OK')
 freeze=ROOT/'frozen_training_sources.json'
 assert freeze.exists(),'Explicitly freeze all training inputs before launch'
 frozen=json.loads(freeze.read_text());completed=[]
 assert not (ROOT/'training_records.json').exists(),'Existing formal run records require audit, not blind restart'
 for job in cohort['formal_training_jobs']:
  assert not (ROOT/'stop').exists()
  for p,h in frozen['files'].items():assert sha(p)==h,p
  parent=cohort['models'][job['parent']];checkpoint=Path(parent['checkpoint']);assert sha(checkpoint)==parent['sha256']
  while shutil.disk_usage(ROOT).free<1500*1024*1024:
   (ROOT/'training_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state='waiting_for_disk_capacity',completed=len(completed),minimum_free_bytes=1500*1024*1024)))
   assert not (ROOT/'stop').exists();time.sleep(20)
  name=f"{job['group']}_s{job['training_seed']}";audit=ROOT/'training'/name;audit.mkdir(parents=True,exist_ok=True)
  args=['python','-u','scripts/locomotion/train.py','--task','Unitree-TW-30dof-Velocity','--headless','--num_envs','4096','--max_iterations','1500','--run_name','issue17_20261113_'+name,'--seed',str(job['training_seed']),'--resume','--load_run',checkpoint.parent.name,'--checkpoint',checkpoint.name,'--experiment_profile','robust08','--issue5_training_mode','strengthen','--issue5_factor','push','--mass_randomization','bounded','--last_action_observation','requested','--post_push_pitch_weight','0.0','--issue12_audit_dir',str(audit),'--issue17_mode',job['group']]
  start=time.time()
  with (audit/'console.log').open('w') as out:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17',*args],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'training_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state='training',name=name,child_pid=child.pid,completed=len(completed),started_unix=start,args=args)))
   while child.poll() is None:
    if (ROOT/'stop').exists():
     os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('Explicit stop requested')
    time.sleep(5)
  assert child.returncode==0,(name,child.returncode)
  summary=json.loads((audit/'run_summary.json').read_text());assert summary['additional_updates']==1500 and summary['environment_steps']==36000 and summary['num_envs']==4096 and not summary['std_reset']
  loaded=json.loads((audit/'loaded_state_audit.json').read_text());assert loaded['passed'] and loaded['checkpoint_sha256']==parent['sha256']
  final=Path(summary['log_dir'])/f"model_{summary['final_iteration']}.pt";assert final.is_file()
  completed.append(dict(**job,checkpoint=str(final),sha256=sha(final),wall_seconds=time.time()-start,summary=summary))
  (ROOT/'training_records.json').write_text(json.dumps(completed,indent=2)+'\n');print('FORMAL_TRAINING_COMPLETED',name,len(completed),'of10',flush=True)
 assert len(completed)==10; (ROOT/'formal_training_complete').write_text('Exactly10x1500updates; final checkpoints only.\n')
if __name__=='__main__':main()
