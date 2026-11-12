"""Follow initial postprocessing to three-view rendering and sampled automatic QA."""
import json,os,time
from pathlib import Path
from issue16_common import ROOT,REPO,run
def main():
 os.chdir(REPO)
 def state(phase,**more):(ROOT/'media_driver.json').write_text(json.dumps(dict(pid=os.getpid(),state=phase,updated_unix=time.time(),**more))+'\n')
 state('waiting_for_initial_postprocessing')
 try:
  while not (ROOT/'initial_postprocessing_complete').exists():
   d=json.loads((ROOT/'postprocess_driver.json').read_text());p=Path('/proc')/str(d['pid'])
   assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Initial postprocessor terminal before completion; inspect before retry'
   assert b'pipeline_issue16.py' in (p/'cmdline').read_bytes(),'Initial postprocessor PID changed identity'
   assert not (ROOT/'stop').exists()
   time.sleep(20)
  for phase,script in [('render_three_views','render_issue16.py'),('video_automatic_QA','audit_issue16_videos.py')]:
   state(phase);run(['python','-u','scripts/locomotion/'+script],ROOT/(phase+'.log'),phase)
  state('complete_manual_review_pending')
  (ROOT/'automated_media_complete').write_text('Three colored clips and21sampled frames checked automatically; actual manual figure/sheet/full-resolution review and final scientific interpretation/publication still required.\n')
  print('ISSUE16_MEDIA_AUTOMATIC_COMPLETE_MANUAL_PENDING',flush=True)
 except Exception as error:
  state('failed',error=repr(error));raise
if __name__=='__main__':main()
