"""Wait for all source states, formal evaluation and E before statistical analysis."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def main():
 def state(s,**extra):(ROOT/'statistics_progress.json').write_text(json.dumps(dict(driver_pid=os.getpid(),state=s,updated_unix=time.time(),**extra)))
 try:
  state('waiting_for_initial_postprocessing')
  while not (ROOT/'initial_postprocessing_complete').exists():
   d=json.loads((ROOT/'postprocess_progress.json').read_text());p=Path('/proc')/str(d['driver_pid'])
   assert p.exists() and (p/'stat').read_text().split()[2]!='Z','Postprocessor stopped before completion; inspect source before any retry'
   assert b'pipeline_issue17.py' in (p/'cmdline').read_bytes();assert not (ROOT/'stop').exists();time.sleep(20)
  outputs={}
  for phase,name in [('all_raw_data_audit','analyze_issue17_raw.py'),('paired_acceptance','aggregate_issue17.py')]:
   source=Path('scripts/locomotion')/name;outputs[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
   with (ROOT/(phase+'.log')).open('w') as log:
    child=subprocess.Popen(['bash','-c','source scripts/locomotion/remote_env.sh; exec "$@"','issue17stats','python','-u',str(source)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state(phase,child_pid=child.pid,log=str(ROOT/(phase+'.log')));assert child.wait()==0,phase
  (ROOT/'statistical_sources_sha256.json').write_text(json.dumps(outputs,indent=2)+'\n');state('statistics_complete_media_and_manual_audit_pending')
  (ROOT/'statistics_complete').write_text('Raw reconstruction and paired acceptance computed; media/manual scientific audit/report/publication still required.\n')
 except Exception as error:state('failed',error=repr(error));raise
if __name__=='__main__':main()
