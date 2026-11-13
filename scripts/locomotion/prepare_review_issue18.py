"""Generate actual review artifacts after rendering; never claims manual review."""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
def state(status,**kw):
 p=ROOT/'review_preparation_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,driver_pid=os.getpid(),updated_at=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def main():
 state('waiting_for_rendered_source_evidence')
 while True:
  d=json.loads((ROOT/'postprocess_progress.json').read_text())
  if d['status']=='analysis_and_render_complete_manual_qa_report_archive_publication_pending':break
  assert d['status']!='failed',d
  p=Path('/proc')/str(d['driver_pid']);assert p.exists() and (p/'stat').read_text().split()[2]!='Z'
  assert b'postprocess_issue18.py' in (p/'cmdline').read_bytes();assert not (ROOT/'stop').exists();time.sleep(20)
 for script in ['audit_issue18_videos.py','plot_issue18.py','report_issue18.py']:
  log=ROOT/(script+'.log')
  with log.open('wb') as out:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue18review','python','-u','scripts/locomotion/'+script],stdout=out,stderr=subprocess.STDOUT)
   state('running',script=script,child_pid=child.pid,log=str(log));assert child.wait()==0,(script,str(log))
 state('actual_manual_visual_review_pending')
if __name__=='__main__':
 try:main()
 except BaseException:state('failed',error=traceback.format_exc());raise
