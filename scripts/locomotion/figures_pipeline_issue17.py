"""CPU completion audit then figures; independent of the GPU source replay queue."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def main():
 def state(s,**extra):(ROOT/'figure_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state=s,updated_unix=time.time(),**extra)))
 def wait(flag,progress,identity):
  while not (ROOT/flag).exists():
   d=json.loads((ROOT/progress).read_text());p=Path('/proc')/str(d['driver_pid']);assert p.exists() and (p/'stat').read_text().split()[2]!='Z';assert identity.encode() in (p/'cmdline').read_bytes();assert not (ROOT/'stop').exists();time.sleep(20)
 def run(name):
  with (ROOT/(name+'.log')).open('w') as out:
   child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17figs','python','-u','scripts/locomotion/'+name],stdout=out,stderr=subprocess.STDOUT,start_new_session=True);state(name,child_pid=child.pid);assert child.wait()==0,name
 try:
  state('waiting_for_training_audit');wait('formal_training_complete','training_progress.json','run_issue17_training.py');run('audit_training_issue17.py')
  state('waiting_for_statistics');wait('statistics_complete','statistics_progress.json','statistics_pipeline_issue17.py');run('plot_issue17.py')
  state('figures_complete_manual_review_pending');(ROOT/'figures_complete').write_text('SixPNG/SVG figures generated; actual visual QA remains.\n')
 except Exception as e:state('failed',error=repr(e));raise
if __name__=='__main__':main()
