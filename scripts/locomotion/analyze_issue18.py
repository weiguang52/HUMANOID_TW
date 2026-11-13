"""Audit reused originals and 45 new jobs without any additional rollout."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from issue15_analysis_data import load_yaw,job_statistics
from issue18_analysis import half_cycles,summary,complete_mean
ROOT=Path('/root/gpufree-data/tw_stage1/issue18_20261113')
COMMANDS=['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']
ZERO=['stand','forward_03','forward_05','forward_08'];ABS='full20s_survivor_heading_mean_abs_degrees'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,d):(ROOT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 progress=read(ROOT/'evaluation_progress.json');assert progress['status']=='completed' and len(progress['completed'])==45
 reused=read(ROOT/'original_reuse_manifest.json')['jobs'];print('original_example_keys',list(reused[0]),flush=True)
 models={m['model']:m for m in read(ROOT/'cohort.json')['models']};jobs=[]
 for j in reused:jobs.append(dict(j,mode='original'))
 for j in read(ROOT/'evaluation_job_templates.json'):
  jobs.append(dict(j,mode='strict',output=str(ROOT/'evaluation/clean'/(j['name']+'.json'))))
 assert len(jobs)==90
 stats={};inputs={};halfdir=ROOT/'half_cycle_raw';halfdir.mkdir(exist_ok=True);csvpath=ROOT/'per_trial_heading.csv.gz';records=0;rows=0;maxerror=0.;fullseries=0;caveats=[]
 fields=['mode','group','training_seed','eval_seed','command','env_id','first_failure_step','survived20s','full20s_signed_degrees','full20s_absolute_degrees','initial_prefix_signed_degrees','initial_prefix_valid_seconds','all_reset_excluded_signed_degrees']
 with gzip.open(csvpath,'wt',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for job in jobs:
   path=Path(job['output']);d=read(path);mode=job['mode'];model=job['model'];ts=models[model]['training_seed'];es=job['eval_seed']
   if mode=='original':
    assert sha(path)==job['output_sha256']
    for p,h in job['source_files_sha256'].items():assert sha(p)==h
   tracking=next(d[k] for k in ['issue18_tracking','issue16_tracking','issue15_tracking'] if k in d)
   folder=Path(tracking['output']);assert tracking['checkpoint_sha256']==models[model]['checkpoint_sha256']
   assert tracking['model_optimizer_state_unchanged']
   assert [x['command'] for x in d['results']]==COMMANDS
   for p in [path,*folder.iterdir()]:
    if p.is_file():inputs[str(p)]=sha(p)
   for result in d['results']:
    cmd=result['command'];z,y,masks,meta=load_yaw(folder,cmd,result);v=job_statistics(z,y,meta,result)
    survived=(y['first_failure_step']<0)&(y['heading_count'][1]==1000)
    degrees=y['heading_delta_sum']*180/np.pi
    for suffix,values in [('signed_mean',degrees[1,survived]),('mean_abs',abs(degrees[1,survived])),('median_abs',abs(degrees[1,survived])),('p95_abs',abs(degrees[1,survived]))]:
     value=None if len(values)==0 else float(np.median(values) if suffix=='median_abs' else np.percentile(values,95) if suffix=='p95_abs' else np.mean(values))
     v['full20s_survivor_heading_'+suffix+'_degrees']=value
    v['full20s_heading_survivors']=int(survived.sum());v['full20s_censored_trials']=int((~survived).sum())
    if 'heading_yaw' in y:
     yaw=y['heading_yaw'].astype(float);prev=np.concatenate([y['heading_start'][None].astype(float),yaw[:-1]]);delta=np.arctan2(np.sin(yaw-prev),np.cos(yaw-prev))
     for scope in [0,1]:
      reconstructed=np.where(masks[:,scope,0],delta,0).sum(0);error=float(abs(reconstructed-y['heading_delta_sum'][scope]).max());maxerror=max(maxerror,error);assert np.allclose(reconstructed,y['heading_delta_sum'][scope],atol=1e-4,rtol=1e-6)
     fullseries+=1
    elif cmd in ZERO:caveats.append(dict(mode=mode,model=model,eval_seed=es,command=cmd,note='Original stand uses preserved all-trial accumulation and independently audited reference trajectories; no all-trial instantaneous heading series was recorded. Original rollout not rerun.'))
    if cmd=='forward_05':
     if mode=='strict':
      with np.load(folder/'forward_05_phase.npz') as ph:phase=ph['pre_action_episode_step']
     else:
      valid=masks[:,0,0];phase=np.zeros((1000,1024),np.int16);counter=np.zeros(1024,np.int16)
      for k in range(1000):phase[k]=counter;counter=np.where(valid[k],counter+1,0)
     h,raw=half_cycles(y['heading_yaw'],y['heading_start'],masks,phase);v.update(h);hp=halfdir/f'{mode}_{model}_eval{es}.npz';np.savez_compressed(hp,**raw)
    key=f'{mode}:{model}:eval{es}:{cmd}';stats[key]=v;records+=1
    for i in range(1024):
     ok=bool(survived[i]);writer.writerow(dict(mode=mode,group=models[model]['group'],training_seed=ts,eval_seed=es,command=cmd,env_id=i,first_failure_step=int(y['first_failure_step'][i]),survived20s=ok,full20s_signed_degrees=float(degrees[1,i]) if ok else None,full20s_absolute_degrees=float(abs(degrees[1,i])) if ok else None,initial_prefix_signed_degrees=float(degrees[1,i]),initial_prefix_valid_seconds=float(y['heading_count'][1,i]*.02),all_reset_excluded_signed_degrees=float(degrees[0,i])));rows+=1
   print('AUDITED',mode,model,es,flush=True)
 assert records==540 and rows==552960
 seeds={};groups={};comparisons={};classifications={}
 for group in ['control6000','control_int','sym_int']:
  for mode in ['original','strict']:
   for cmd in COMMANDS:
    key=f'{mode}:{group}:{cmd}';seedvalues=[]
    for ts in range(1,6):
     values=[stats[f'{mode}:{group}_s{ts}:eval{es}:{cmd}'] for es in [1,2,3]]
     seedvalues.append({k:complete_mean([v[k] for v in values]) for k in values[0]})
    seeds[key]=seedvalues;groups[key]={k:summary([v[k] for v in seedvalues]) for k in seedvalues[0]}
  for cmd in COMMANDS:
   o=seeds[f'original:{group}:{cmd}'];p=seeds[f'strict:{group}:{cmd}'];comparisons[f'{group}:{cmd}']={k:summary([None if x[k] is None or y[k] is None else y[k]-x[k] for x,y in zip(o,p)]) for k in o[0]}
  fall={cmd:comparisons[f'{group}:{cmd}']['fall_rate']['mean'] for cmd in ZERO};interpretable=all(x is not None and x<=.05 for x in fall.values());orig=groups[f'original:{group}:forward_05'][ABS]['mean'];proj=groups[f'strict:{group}:forward_05'][ABS]['mean'];ci=comparisons[f'{group}:forward_05'][ABS]['ci95'];headings={cmd:groups[f'strict:{group}:{cmd}'][ABS]['mean'] for cmd in ZERO};small=sum(x is not None and x<=30 for x in headings.values());half=groups[f'strict:{group}:forward_05']['complete_cycle_sum_mean_degrees'];hc=half['ci95'];A=interpretable and orig is not None and proj is not None and proj<=.30*orig and ci is not None and ci[1]<0 and small>=3;B=interpretable and not A and ((proj is not None and proj>100) or (hc is not None and (hc[0]>0 or hc[1]<0)))
  classifications[group]=dict(interpretable=interpretable,fall_gain_by_zero_yaw_command=fall,original_forward05_abs_heading=orig,strict_forward05_abs_heading=proj,ratio=None if orig in [None,0] or proj is None else proj/orig,paired_absolute_heading_difference_ci95=ci,strict_zero_yaw_heading_means=headings,commands_at_most30deg=small,complete_cycle_sum=half,classification='A' if A else 'B' if B else 'C' if interpretable else 'wrapper_destroyed_control')
 usable=[x for x in classifications.values() if x['interpretable']];overall='A' if len(usable)>=2 and all(x['classification']=='A' for x in usable) else 'B' if usable and all(x['classification']=='B' for x in usable) else 'C'
 result=dict(overall=overall,cohorts=classifications,no_causal_conclusion=not bool(usable),prospective_rules_sha256=sha(ROOT/'classification_rules.json'),conservative_overall_mixed_cohort_convention_disclosed=True)
 for name,obj in [('job_statistics.json',stats),('seed_statistics.json',seeds),('group_statistics.json',groups),('paired_comparisons.json',comparisons),('classification.json',result)]:save(name,obj)
 save('raw_analysis_audit.json',dict(passed=True,total_jobs=90,new_jobs=45,reused_original_jobs=45,command_records=records,per_trial_rows=rows,all_trial_instantaneous_heading_series=fullseries,maximum_reconstruction_error_radians=maxerror,original_series_limitations=caveats,raw_inputs_sha256=inputs,per_trial_csv_sha256=sha(csvpath),zero_training=True,no_original_rollouts_rerun=True,source_sha256=sha(__file__)))
 print('ISSUE18_ANALYSIS_PASS',overall,records,rows,flush=True)
if __name__=='__main__':main()
