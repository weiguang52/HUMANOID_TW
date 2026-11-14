"""Continue issue19 through final exports, analysis and media, stopping for actual review."""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
REPO=Path('/root/gpufree-data/projects/HUMANOID_TW')
def state(phase,**kw):
 (ROOT/'postprocess_progress.json').write_text(json.dumps(dict(phase=phase,driver_pid=os.getpid(),updated=time.time(),**kw),indent=2))
def waitfor(name):
 while not (ROOT/name).exists():
  assert not (ROOT/'stop').exists()
  state('waiting_for_'+name);time.sleep(20)
def run(script):
 assert not (ROOT/'stop').exists()
 with (ROOT/(script+'.log')).open('w') as log:
  p=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec python -u "$1"','issue19post','scripts/locomotion/'+script+'.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state(script,child_pid=p.pid)
  assert p.wait()==0,(script,p.returncode)
 print('ISSUE19_POSTPROCESS_STAGE_DONE',script,flush=True)
def main():
 os.chdir(REPO)
 try:
  waitfor('formal_training_complete')
  run('validate_issue19_final')
  waitfor('formal_evaluation_complete')
  for script in ['analyze_issue19','plot_issue19','report_issue19','select_video_issue19','reconstruct_issue19','render_issue19','audit_issue19_videos']:run(script)
  state('actual_manual_plot_and_video_review_required')
  print('ISSUE19_AUTOMATIC_ANALYSIS_AND_MEDIA_DONE_MANUAL_REVIEW_REQUIRED',flush=True)
 except BaseException:
  state('failed',error=traceback.format_exc());raise
if __name__=='__main__':main()
