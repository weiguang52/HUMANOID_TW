"""Follow exact formal training to sequential state collection, evaluation and offline E."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def main():
 os.chdir(REPO)
 def state(phase,**extra):(ROOT/'postprocess_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state=phase,updated_unix=time.time(),**extra)))
 def verify():
  for manifest in ['frozen_training_sources.json','frozen_evaluation_sources.json']:
   for p,h in json.loads((ROOT/manifest).read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 try:
  state('waiting_for_formal_training')
  while not (ROOT/'formal_training_complete').exists():
   d=json.loads((ROOT/'training_progress.json').read_text());p=Path('/proc')/str(d['driver_pid'])
   assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Training driver terminated before completion; inspect before any retry'
   assert b'run_issue17_training.py' in (p/'cmdline').read_bytes()
   assert not (ROOT/'stop').exists();time.sleep(20)
  verify()
  for phase,script in [('fixed_state_collection','collect_states_issue17.py'),('formal_evaluation','run_issue17_evaluation.py'),('offline_action_equivariance','equivariance_issue17.py')]:
   verify()
   with (ROOT/(phase+'_driver.log')).open('w') as log:
    child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17pipeline','python','-u','scripts/locomotion/'+script],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state(phase,child_pid=child.pid,log=str(ROOT/(phase+'_driver.log')))
    assert child.wait()==0,phase
  verify();state('statistics_media_manual_review_pending')
  (ROOT/'initial_postprocessing_complete').write_text('Fixed state set,90formal evals and offline E complete; statistics/media/manualQA/publication remain.\n')
 except Exception as error:state('failed',error=repr(error));raise
if __name__=='__main__':main()
