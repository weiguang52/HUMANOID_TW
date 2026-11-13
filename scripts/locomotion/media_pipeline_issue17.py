"""Generate exact-state new-group media only after all formal statistics exist."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def main():
 def state(s,**more):(ROOT/'media_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state=s,updated_unix=time.time(),**more)))
 try:
  state('waiting_for_statistics')
  while not (ROOT/'statistics_complete').exists():
   d=json.loads((ROOT/'statistics_progress.json').read_text());p=Path('/proc')/str(d['driver_pid'])
   assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Statistics driver terminal before completion; inspect before retry'
   assert b'statistics_pipeline_issue17.py' in (p/'cmdline').read_bytes();assert not (ROOT/'stop').exists();time.sleep(20)
  scripts=['select_video_issue17.py','reconstruct_issue17.py','render_issue17.py','audit_issue17_videos.py']
  sources={str((Path('scripts/locomotion')/n).resolve()):hashlib.sha256((Path('scripts/locomotion')/n).read_bytes()).hexdigest() for n in [*scripts,'replay_issue17.py','render_eval_issue17.py']}
  (ROOT/'media_execution_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
  for name in scripts:
   for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
   assert not (ROOT/'stop').exists()
   with (ROOT/(name+'.log')).open('w') as log:
    child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17media','python','-u','scripts/locomotion/'+name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state(name,child_pid=child.pid,log=str(ROOT/(name+'.log')));assert child.wait()==0,name
  state('automatic_media_complete_manual_review_pending');(ROOT/'automatic_media_complete').write_text('Two source-bound colored20s clips and14sample QA complete. Actual manual critical-frame/figure review and publication remain.\n')
 except Exception as error:state('failed',error=repr(error));raise
if __name__=='__main__':main()
