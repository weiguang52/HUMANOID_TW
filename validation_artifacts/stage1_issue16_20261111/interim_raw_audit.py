"""Read-only interim reconstruction; never a final scientific comparison."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,'scripts/locomotion')
from issue16_common import ROOT,sha
from analyze_issue16 import load_yaw,independent_series
def main():
 receipts=json.loads((ROOT/'formal_job_audits.json').read_text())
 jobs={j['name']:j for j in json.loads((ROOT/'formal_jobs.json').read_text())}
 count=0;series=0;max_error=0.;inputs={};names=[]
 for receipt in receipts:
  assert receipt['passed'];job=jobs[receipt['name']];path=Path(job['output'])
  assert sha(path)==receipt['output_sha256'];inputs[str(path)]=sha(path)
  d=json.loads(path.read_text());folder=Path(d['issue16_tracking']['output'])
  for r in d['results']:
   z,y,masks,meta=load_yaw(folder,r['command'],r);error=independent_series(z,y,masks,meta);count+=1
   if error is not None:series+=1;max_error=max(max_error,error)
   for suffix in ['.npz','_yaw.npz','.json']:
    p=folder/(r['command']+suffix)
    if p.is_file():inputs[str(p)]=sha(p)
  names.append(job['name'])
 result=dict(passed=True,interim_only=True,formal_evaluation_complete=False,no_scientific_conclusion=True,
  new_trials=0,completed_jobs_audited=len(names),completed_job_names=names,
  command_records_audited=count,all_trial_heading_arrays_audited=series,
  maximum_heading_accumulation_error_radians=max_error,inputs_sha256=inputs,source_sha256=sha(__file__),checked_unix=time.time())
 out=ROOT/('interim_raw_audit_'+str(len(names))+'_jobs.json')
 out.write_text(json.dumps(result,indent=2)+'\n')
 print('INTERIM_RAW_AUDIT_PASSED',len(names),count,series,max_error,flush=True)
if __name__=='__main__':main()
