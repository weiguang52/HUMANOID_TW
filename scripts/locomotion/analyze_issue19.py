"""Audit all issue19 formal records and reuse controls without extra rollout."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from issue15_analysis_data import load_yaw,job_statistics
from issue18_analysis import half_cycles,summary,complete_mean
from issue19_analysis import decompose,summarize_decomposition
ROOT=Path('/root/gpufree-data/tw_stage1/issue19_20261113')
CLEAN=['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']
ZERO=['stand','forward_03','forward_05','forward_08']
ABS='full20s_survivor_heading_mean_abs_degrees'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
 assert (ROOT/'formal_evaluation_complete').exists()
 new=read(ROOT/'evaluation_records.json');assert len(new)==60
 refs=read(ROOT/'reference_reuse_manifest.json')['jobs'];assert len(refs)==45
 stats={};input_hashes={};record_count=0;trial_count=0;limitations=[]
 folder=ROOT/'decomposition_raw';folder.mkdir(exist_ok=True)
 fields=['group','training_seed','eval_seed','protocol','command','env_id','survived20s','first_failure_step','full20s_signed_degrees','full20s_absolute_degrees','initial_prefix_signed_degrees','initial_prefix_seconds','startup_signed_degrees','startup_prefix_signed_degrees','startup_prefix_seconds','late_full20s_slope_degrees_per_second','late_initial_prefix_slope_degrees_per_second','late_prefix_samples']
 csvpath=ROOT/'per_trial_heading_decomposition.csv.gz'
 with gzip.open(csvpath,'wt',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for job in [*new,*refs]:
   path=Path(job['output']);assert sha(path)==job['output_sha256']
   for p,h in job.get('source_files_sha256',{}).items():assert sha(p)==h,p
   d=read(path);tracking=next(d[k] for k in ['issue18_tracking','issue16_tracking'] if k in d)
   assert tracking['model_optimizer_state_unchanged']
   moment=Path(tracking['output']);assert [x['command'] for x in d['results']]==job['commands']
   input_hashes[str(path)]=sha(path)
   for p in moment.iterdir():
    if p.is_file():input_hashes[str(p)]=sha(p)
   for result in d['results']:
    cmd=result['command'];z,y,masks,meta=load_yaw(moment,cmd,result)
    assert meta['num_envs']==1024 and meta['steps']==1000
    v=job_statistics(z,y,meta,result)
    survived=(y['first_failure_step']<0)&(y['heading_count'][1]==1000)
    degrees=y['heading_delta_sum'][1]*180/np.pi
    v[ABS]=float(np.abs(degrees[survived]).mean()) if survived.any() else None
    v['full20s_censored_trials']=int((~survived).sum())
    decomposition=None
    if 'heading_yaw' in y:
     decomposition=decompose(y['heading_yaw'],y['heading_start'],masks)
     assert np.array_equal(decomposition['survived20s'],survived)
     assert np.allclose(decomposition['initial_prefix_signed_degrees'],degrees,atol=.006,rtol=1e-6)
     v.update(summarize_decomposition(decomposition))
    else:limitations.append(dict(group=job['group'],protocol=job['protocol'],training_seed=job['training_seed'],eval_seed=job['eval_seed'],command=cmd,reason='Historical reference has all-trial accumulated heading but no instantaneous yaw for this command; no reference rollout rerun.'))
    raw={} if decomposition is None else dict(decomposition)
    if cmd in ['forward_05','forward_08']:
     phasepath=moment/(cmd+'_phase.npz')
     if phasepath.exists():
      with np.load(phasepath) as p:phase=p['pre_action_episode_step']
     else:
      phase=np.zeros((1000,1024),np.int16);counter=np.zeros(1024,np.int16)
      for k in range(1000):phase[k]=counter;counter=np.where(masks[k,0,0],counter+1,0)
     h,halfraw=half_cycles(y['heading_yaw'],y['heading_start'],masks,phase);v.update(h);raw.update(halfraw)
    stem=job['protocol']+'_'+job['group']+'_s'+str(job['training_seed'])+'_eval'+str(job['eval_seed'])+'_'+cmd
    if raw:np.savez_compressed(folder/(stem+'.npz'),**raw)
    key=':'.join([job['protocol'],job['group'],'s'+str(job['training_seed']),'eval'+str(job['eval_seed']),cmd])
    assert key not in stats;stats[key]=v
    for i in range(1024):
     row=dict(group=job['group'],training_seed=job['training_seed'],eval_seed=job['eval_seed'],protocol=job['protocol'],command=cmd,env_id=i,survived20s=bool(survived[i]),first_failure_step=int(y['first_failure_step'][i]),full20s_signed_degrees=float(degrees[i]) if survived[i] else None,full20s_absolute_degrees=float(abs(degrees[i])) if survived[i] else None,initial_prefix_signed_degrees=float(degrees[i]),initial_prefix_seconds=float(y['heading_count'][1,i]*.02))
     if decomposition is not None:
      for k in fields:
       if k in decomposition:
        value=decomposition[k][i];row[k]=bool(value) if isinstance(value,(bool,np.bool_)) else float(value) if np.isfinite(value) else None
     writer.writerow(row);trial_count+=1
    record_count+=1
   print('ISSUE19_ANALYZED',job['name'],flush=True)
 assert record_count==495 and trial_count==506880
 seeds={};groups={}
 for group,protocols in [('sym_wrap',['clean','combined']),('sym_cont',['clean','combined']),('control_int',['clean','combined']),('sym_initial_strict',['clean'])]:
  for protocol in protocols:
   for cmd in CLEAN if protocol=='clean' else ['forward_03','forward_05','forward_08']:
    seedvalues=[]
    for ts in range(1,6):
     values=[stats[':'.join([protocol,group,'s'+str(ts),'eval'+str(es),cmd])] for es in [1,2,3]]
     keys=set.intersection(*(set(v) for v in values))
     seedvalues.append({k:complete_mean([v[k] for v in values]) for k in sorted(keys)})
    key=':'.join([protocol,group,cmd]);seeds[key]=seedvalues
    keys=set.intersection(*(set(v) for v in seedvalues))
    groups[key]={k:summary([v[k] for v in seedvalues]) for k in sorted(keys)}
 comparisons={}
 for target in ['sym_cont','control_int','sym_initial_strict']:
  for protocol in ['clean'] if target=='sym_initial_strict' else ['clean','combined']:
   for cmd in CLEAN if protocol=='clean' else ['forward_03','forward_05','forward_08']:
    a=seeds[':'.join([protocol,'sym_wrap',cmd])];b=seeds[':'.join([protocol,target,cmd])]
    keys=set.intersection(*(set(v) for v in [*a,*b]))
    comparisons[':'.join([protocol,'sym_wrap-minus-'+target,cmd])]={k:summary([None if x[k] is None or y[k] is None else x[k]-y[k] for x,y in zip(a,b)]) for k in sorted(keys)}
 def metric(protocol,group,cmd,key):return groups[':'.join([protocol,group,cmd])][key]
 def mean(protocol,group,cmd,key):return metric(protocol,group,cmd,key)['mean']
 def atmost(a,b):return a is not None and b is not None and a<=b
 def atleast(a,b):return a is not None and b is not None and a>=b
 heading={cmd:metric('clean','sym_wrap',cmd,ABS) for cmd in ZERO}
 ci=comparisons['clean:sym_wrap-minus-sym_cont:forward_05'][ABS]['ci95']
 primary=ci is not None and ci[1]<0 and atmost(heading['forward_05']['mean'],30)
 allheading=all(atmost(x['mean'],30) for x in heading.values())
 tracking={}
 for cmd in ['forward_05','forward_08']:
  xy=mean('clean','sym_wrap',cmd,'full_xy_error');refxy=mean('clean','control_int',cmd,'full_xy_error')
  vx=mean('clean','sym_wrap',cmd,'full_vx_mean');refvx=mean('clean','control_int',cmd,'full_vx_mean')
  tracking[cmd]=dict(xy_error=xy,reference_xy_error=refxy,xy_error_ratio=None if xy is None or refxy in [None,0] else xy/refxy,vx=vx,reference_vx=refvx,vx_ratio=None if vx is None or refvx in [None,0] else vx/refvx,passed=atmost(xy,None if refxy is None else 1.05*refxy) and atleast(vx,None if refvx is None else .95*refvx))
 for cmd in ['yaw_positive','yaw_negative']:
  error=mean('clean','sym_wrap',cmd,'full_yaw_error');ref=mean('clean','control_int',cmd,'full_yaw_error')
  tracking[cmd]=dict(yaw_error=error,reference_yaw_error=ref,ratio=None if error is None or ref in [None,0] else error/ref,passed=atmost(error,None if ref is None else 1.10*ref))
 fall03=metric('combined','sym_wrap','forward_03','fall_rate')
 fall05=mean('combined','sym_wrap','forward_05','fall_rate');reffall05=mean('combined','control_int','forward_05','fall_rate')
 robust=fall03['ci95'] is not None and fall03['ci95'][1]<.01 and atmost(fall05,reffall05)
 recovered=all(x['passed'] for x in tracking.values())
 passed=primary and allheading and recovered and robust
 heading_passed=primary and allheading
 diagnostic={}
 for cmd in ['forward_05','forward_08']:
  slope=metric('clean','sym_wrap',cmd,'late_full20s_slope_degrees_per_second_mean')
  half=metric('clean','sym_wrap',cmd,'complete_cycle_sum_mean_degrees')
  startup=metric('clean','sym_wrap',cmd,'startup_signed_degrees_mean')
  nonzero=lambda x:x['ci95'] is not None and (x['ci95'][0]>0 or x['ci95'][1]<0)
  diagnostic[cmd]=dict(startup=startup,late_slope=slope,half_cycle_sum=half,persistent_and_nonzero_halfcycle=nonzero(slope) and nonzero(half))
 persistent=any(x['persistent_and_nonzero_halfcycle'] for x in diagnostic.values())
 if passed:outcome='all_acceptance_criteria_met';recommendation='No additional experiment in this issue; review deployment eligibility separately.'
 elif heading_passed and not recovered:outcome='heading_solved_tracking_not_recovered';recommendation='Next issue: address tracking recovery alone, keeping the symmetric actor.'
 elif heading_passed:outcome='heading_met_other_acceptance_not_met';recommendation='Next issue: address bounded-combined robustness alone.'
 elif persistent:outcome='heading_not_met_persistent_drift_and_nonzero_halfcycle';recommendation='Next issue: original/reflected policy x original/strict symmetric USD 2x2 diagnostic.'
 else:outcome='heading_not_met_startup_or_inconclusive';recommendation='Next issue: test randomized initial gait phase alone; current decomposition does not establish a physical cause.'
 conclusion=dict(all_criteria_passed=passed,primary_passed=primary,zero_yaw_heading_passed=allheading,heading_passed=heading_passed,tracking_recovered=recovered,robustness_passed=robust,outcome=outcome,recommendation=recommendation,recommendation_not_implemented=True,default_promoted=False,primary_paired_difference_ci95=ci,heading_by_command=heading,tracking=tracking,bounded03_falls=fall03,bounded05_falls=fall05,reference_bounded05_falls=reffall05,diagnostic=diagnostic,rules_sha256=sha(ROOT/'classification_rules.json'),bootstrap_unit='five paired training seeds, three evaluation seeds averaged within each',heading_seed95_intervals='individual percentile intervals; not adjusted for simultaneous coverage',causal_background_claims_not_proven=True)
 for name,obj in [('job_statistics.json',stats),('seed_statistics.json',seeds),('group_statistics.json',groups),('paired_comparisons.json',comparisons),('scientific_conclusion.json',conclusion)]:save(name,obj)
 save('raw_analysis_audit.json',dict(passed=True,new_jobs=60,reused_reference_jobs=45,total_jobs=105,command_records=record_count,initial_trial_rows=trial_count,raw_inputs_sha256=input_hashes,historical_series_limitations=limitations,per_trial_csv_sha256=sha(csvpath),source_sha256=sha(__file__)))
 print('ISSUE19_SCIENTIFIC_ANALYSIS_COMPLETE',outcome,flush=True)
if __name__=='__main__':main()
