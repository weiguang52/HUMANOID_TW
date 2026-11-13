"""Independent reconstruction of all450records and460800initial-trial rows."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from issue15_analysis_data import load_yaw,job_statistics
from analyze_issue16 import independent_series
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (ROOT/'formal_evaluation_complete').exists();jobs=json.loads((ROOT/'evaluation_records.json').read_text());assert len(jobs)==90
 stats={};inputs={};commands=0;arrays=0;error=0.;rows=0;target=ROOT/'per_trial_heading.csv.gz'
 fields=['group','training_seed','eval_seed','protocol','command','env_id','first_failure_step','survived20s','full20s_heading_signed_degrees','full20s_heading_absolute_degrees','prefix_heading_signed_degrees','prefix_valid_seconds','all_reset_excluded_heading_signed_degrees']
 with gzip.open(target,'wt',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
  for job in jobs:
   path=Path(job['output']);assert sha(path)==job['output_sha256'];d=json.loads(path.read_text());folder=Path(d['issue16_tracking']['output']);assert d['issue17_clock']['integer_enabled'] and d['issue16_tracking']['model_optimizer_state_unchanged']
   for p in [path,*folder.iterdir()]:
    if p.is_file():inputs[str(p)]=sha(p)
   for result in d['results']:
    cmd=result['command'];z,y,masks,meta=load_yaw(folder,cmd,result);check=independent_series(z,y,masks,meta)
    if check is not None:arrays+=1;error=max(error,check)
    key=f"{job['protocol']}:{job['model']}:eval{job['eval_seed']}:{cmd}";v=job_statistics(z,y,meta,result);v.update(legacy_xy_error=result['metrics']['xy_error'],legacy_yaw_error=result['metrics']['yaw_error']);stats[key]=v;commands+=1
    heading=y['heading_delta_sum']*180/np.pi;failure=y['first_failure_step']
    for i in range(1024):
     ok=failure[i]<0
     writer.writerow(dict(group=job['group'],training_seed=job['training_seed'],eval_seed=job['eval_seed'],protocol=job['protocol'],command=cmd,env_id=i,first_failure_step=int(failure[i]),survived20s=bool(ok),full20s_heading_signed_degrees=float(heading[0,i]) if ok else None,full20s_heading_absolute_degrees=float(abs(heading[0,i])) if ok else None,prefix_heading_signed_degrees=float(heading[1,i]),prefix_valid_seconds=float(y['heading_count'][1,i]*.02),all_reset_excluded_heading_signed_degrees=float(heading[0,i])))
     rows+=1
   print('RAW_JOB_AUDITED',job['name'],flush=True)
 assert commands==450 and arrays==330 and rows==460800
 (ROOT/'job_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
 audit=dict(passed=True,formal_jobs=90,command_records=commands,initial_trial_rows=rows,all_trial_heading_arrays=arrays,max_independent_heading_error_radians=error,all_raw_inputs_sha256=inputs,per_trial_csv_sha256=sha(target),job_statistics_sha256=sha(ROOT/'job_statistics.json'),source_sha256=sha(__file__),no_censored_seed_dropped=True)
 (ROOT/'raw_analysis_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('ISSUE17_ALL_RAW_AUDIT_PASSED',commands,rows,arrays,flush=True)
if __name__=='__main__':main()
