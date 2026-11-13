"""Follow the one bounded evaluation queue, then analyze and render its evidence."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
def read(p):return json.loads(Path(p).read_text())
def state(status,**kw):
 p=ROOT/'postprocess_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,driver_pid=os.getpid(),updated_at=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def main():
 state('waiting_for_formal_queue')
 while True:
  d=read(ROOT/'evaluation_progress.json')
  if d['status']=='completed':break
  assert d['status']!='failed',d
  p=Path('/proc')/str(d['driver_pid']);assert p.exists() and (p/'stat').read_text().split()[2]!='Z','queue terminal without completion; inspect actual job before retry'
  assert b'run_issue18_evaluations.py' in (p/'cmdline').read_bytes();assert not (ROOT/'stop').exists();time.sleep(20)
 scripts=['analyze_issue18.py','select_video_issue18.py','reconstruct_issue18.py','render_issue18.py']
 names=[*scripts,'issue18_analysis.py','replay_issue18.py','render_eval_issue18.py']
 sources={str(Path('scripts/locomotion',n).resolve()):hashlib.sha256(Path('scripts/locomotion',n).read_bytes()).hexdigest() for n in names}
 (ROOT/'postprocess_execution_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 for script in scripts:
  for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
  assert not (ROOT/'stop').exists()
  log=ROOT/(script+'.log')
  with log.open('wb') as out:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue18postprocess','python','-u','scripts/locomotion/'+script],stdout=out,stderr=subprocess.STDOUT)
   state('running',script=script,child_pid=child.pid,log=str(log));assert child.wait()==0,(script,str(log))
 state('analysis_and_render_complete_manual_qa_report_archive_publication_pending')
if __name__=='__main__':
 try:main()
 except BaseException:state('failed',error=traceback.format_exc());raise
