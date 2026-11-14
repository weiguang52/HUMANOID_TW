"""Run independent numerical and full-pair audits after all prescribed evaluation cells."""
import os,json,time,subprocess
from pathlib import Path
from issue20_common import ROOT,REPO,sha,save

def main():
 os.chdir(REPO);sources={str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['audit_issue20_incremental.py','audit_issue20_statistics.py','bind_issue20_references.py']};save('frozen_verification_sources.json',{'files':sources,'frozen_before_aggregate_analysis':not (ROOT/'symmetric_analysis.json').exists()})
 while not (ROOT/'formal_evaluation_complete').exists():
  assert Path('/proc/120504').exists(),'Evaluation driver terminal before completion; inspect actual failure'
  save('verification_progress.json',{'state':'waiting_for_formal_evaluation','driver_pid':os.getpid()});time.sleep(20)
 for script in ['audit_issue20_incremental.py','audit_issue20_statistics.py','bind_issue20_references.py']:
  for f,digest in sources.items():assert sha(f)==digest,f
  with (ROOT/(script.removesuffix('.py')+'.log')).open('w') as log:
   child=subprocess.Popen(['python','-u','scripts/locomotion/'+script],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save('verification_progress.json',{'state':script,'driver_pid':os.getpid(),'child_pid':child.pid});assert child.wait()==0,script
 save('verification_progress.json',{'state':'independent_checks_complete','driver_pid':os.getpid()});print('ISSUE20_INDEPENDENT_VERIFICATION_COMPLETE',flush=True)
if __name__=='__main__':main()
