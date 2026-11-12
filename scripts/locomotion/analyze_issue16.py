"""Issue16 raw-data audit, paired seed statistics and exact heading-median selection."""
import csv,gzip,hashlib,itertools,json,os,subprocess
from pathlib import Path
import numpy as np
from issue16_common import ROOT,REPO,sha
from run_issue16 import verify_frozen,audit_job
from issue15_analysis_data import load_yaw,job_statistics
from issue15_math import paired_summary
MAIN=['stand','forward_03','forward_05','forward_08']
ABS='full20s_survivor_heading_mean_abs_degrees'
SIGNED='full20s_survivor_heading_signed_mean_degrees'
A=REPO/'validation_artifacts/stage1_issue15_20261111'
SOURCE_SHA=sha(__file__)

def mean_complete(values):
 return None if any(v is None for v in values) else float(np.mean(values))
def summary(values):
 return None if any(v is None for v in values) else paired_summary(values)
def independent_series(z,y,masks,meta):
 if not meta['command'] in MAIN:return None
 n=meta['num_envs'];heading=y['heading_yaw'].astype('float64');wz=y['wz'].astype('float64')
 assert heading.shape==wz.shape==(1000,n)
 prev=np.concatenate([y['heading_start'][None].astype('float64'),heading[:-1]])
 delta=np.arctan2(np.sin(heading-prev),np.cos(heading-prev))
 error=0.
 for scope in [0,1]:
  valid=masks[:,scope,0]
  recomputed=np.where(valid,delta,0).sum(0)
  error=max(error,float(abs(recomputed-y['heading_delta_sum'][scope]).max()))
  assert np.allclose(recomputed,y['heading_delta_sum'][scope],atol=1e-4,rtol=1e-6)
  assert np.allclose(np.where(valid,wz,0).sum(0),z['sums'][scope,0,:,4],atol=1e-8,rtol=1e-10)
  assert np.allclose(np.where(valid,abs(wz),0).sum(0),z['sums'][scope,0,:,1],atol=1e-8,rtol=1e-10)
 return error

def select_original_representative(cohort):
 candidates=[];original=json.loads((A/'cohort.json').read_text());selection_inputs={}
 manifest=json.loads((A/'sha256.json').read_text())
 assert (A/'sha256.json').read_bytes()==subprocess.check_output(['git','show','770014ee9a4393da6dff34eced89a04d83e788dd:'+str((A/'sha256.json').relative_to(REPO))])
 for name in original['nodes']['control6000']:
  m=original['models'][name]
  for es in [1,2,3]:
   p=A/'clean'/f"{m['evaluation_model']}_eval{es}.json";d=json.loads(p.read_text())
   folder=A/'clean/moments'/f"{m['evaluation_model']}_eval{es}"
   for source in [p,folder/'forward_05_yaw.npz',folder/'forward_05.npz']:
    h=sha(source);assert h==manifest[str(source.relative_to(A))],str(source)
    selection_inputs[str(source)]=h
   with np.load(folder/'forward_05_yaw.npz') as z:
    drift=z['heading_delta_sum'][0]*180/np.pi;failure=z['first_failure_step']
   assert (failure<0).all(),'Existing source has censored trials; do not silently filter the requested median'
   with np.load(folder/'forward_05.npz') as z:
    yaw_error=z['sums'][0,0,:,1]/z['counts'][0,0]
   candidates += [dict(model=name,training_seed=m['training_seed'],eval_seed=es,env_id=i,
    heading_abs_degrees=float(abs(v)),heading_signed_degrees=float(v),yaw_error=float(yaw_error[i]),checkpoint=m['checkpoint'],checkpoint_sha256=m['sha256'],original_job=str(p))
    for i,v in enumerate(drift)]
 assert len(candidates)==15360
 median=float(np.median([v['heading_abs_degrees'] for v in candidates]))
 best=min(candidates,key=lambda x:(abs(x['heading_abs_degrees']-median),x['training_seed'],x['eval_seed'],x['env_id']))
 best.update(selection_inputs_sha256=selection_inputs,committed_archive_manifest_sha256=sha(A/'sha256.json'),command='forward_05',group='control6000',group_median_absolute_heading_degrees=median,candidate_count=len(candidates),
             survival_filter=False,all_original_initial_trials_happen_to_survive=True,selection_metric='absolute reset-aware quaternion20s heading drift; not body-z yaw MAE')
 (ROOT/'representative_selection.json').write_text(json.dumps(best,indent=2)+'\n')
 return best

def main():
 os.chdir(REPO);verify_frozen();assert (ROOT/'formal_evaluation_complete').exists()
 cohort=json.loads((ROOT/'cohort.json').read_text());jobs=json.loads((ROOT/'formal_jobs.json').read_text())
 assert len(jobs)==34
 baseline_seed=json.loads((A/'seed_statistics.json').read_text());baseline_nodes=json.loads((A/'node_statistics.json').read_text())
 stats={};raw_inputs={};recomputed=0;full_series=0;max_heading_error=0.;total_trials=0
 path=ROOT/'per_trial_heading.csv.gz'
 fields=['mode','model','group','training_seed','eval_seed','command','env_id','first_failure_step','survived20s','full20s_heading_signed_degrees','full20s_heading_absolute_degrees','initial_prefix_heading_signed_degrees','initial_prefix_heading_absolute_degrees','initial_prefix_duration_s','all_reset_excluded_segments_heading_signed_degrees','full_valid_samples','body_yaw_MAE_rad_s']
 with gzip.open(path,'wt',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  csv_rows=0
  for job in jobs:
   audit_job(job,cohort);d=json.loads(Path(job['output']).read_text());model=cohort['models'][job['model']]
   folder=Path(d['issue16_tracking']['output'])
   for p in [Path(job['output']),*folder.iterdir()]:
    if p.is_file() and p.suffix in ['.npz','.json','.yaml']:raw_inputs[str(p)]=sha(p)
   for row in d['results']:
    cmd=row['command'];z,y,masks,meta=load_yaw(folder,cmd,row);value=job_statistics(z,y,meta,row)
    error=independent_series(z,y,masks,meta)
    if error is not None:full_series+=1;max_heading_error=max(max_heading_error,error)
    key=f"{job['mode']}:{job['model']}:eval{job['seed']}:{cmd}";stats[key]=value
    recomputed+=1;total_trials+=1024
    degrees=y['heading_delta_sum']*180/np.pi;failed=y['first_failure_step'];c=z['counts']
    for i in range(1024):
     ok=failed[i]<0
     writer.writerow(dict(mode=job['mode'],model=job['model'],group=model['node'] if job['mode']=='mirror' else 'no_policy_default_PD',
      training_seed=model['training_seed'] if job['mode']=='mirror' else None,eval_seed=job['seed'],command=cmd,env_id=i,
      first_failure_step=int(failed[i]),survived20s=bool(ok),full20s_heading_signed_degrees=float(degrees[0,i]) if ok else None,
      full20s_heading_absolute_degrees=float(abs(degrees[0,i])) if ok else None,initial_prefix_heading_signed_degrees=float(degrees[1,i]),
      initial_prefix_heading_absolute_degrees=float(abs(degrees[1,i])),initial_prefix_duration_s=float(c[1,0,i]*.02),
      all_reset_excluded_segments_heading_signed_degrees=float(degrees[0,i]),full_valid_samples=int(c[0,0,i]),
      body_yaw_MAE_rad_s=float(z['sums'][0,0,i,1]/c[0,0,i]) if c[0,0,i] else None))
     csv_rows+=1
   print('AUDITED',job['name'],flush=True)
 assert recomputed==199 and total_trials==csv_rows==203776 and full_series==133
 seeds={};groups={};comparisons=[]
 for node,names in cohort['nodes'].items():
  seeds[node]={};groups[node]={}
  for cmd in ['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']:
   byseed={}
   for name in names:
    ts=str(cohort['models'][name]['training_seed']);values=[stats[f"mirror:{name}:eval{es}:{cmd}"] for es in [1,2,3]]
    byseed[ts]={k:mean_complete([r[k] for r in values]) for k in values[0]}
   seeds[node][cmd]=byseed
   groups[node][cmd]=dict(training_seeds=len(byseed),seed_ids=sorted(byseed,key=int),
      metrics={k:summary([r[k] for r in byseed.values()]) for k in next(iter(byseed.values()))})
   ids=sorted(byseed,key=int);original=[baseline_seed[node][cmd][k] for k in ids];mirror=[byseed[k] for k in ids]
   original_signed=[r[SIGNED] for r in original];mirror_signed=[r[SIGNED] for r in mirror];oa=[r[ABS] for r in original];ma=[r[ABS] for r in mirror]
   finite=all(v is not None for v in [*original_signed,*mirror_signed,*oa,*ma])
   record=dict(node=node,command=cmd,primary_zero_yaw_command=cmd in MAIN,training_seed_ids=ids,
    original=baseline_nodes[node][cmd],mirrored=groups[node][cmd],per_seed=[dict(training_seed=k,original_signed=original[i][SIGNED],mirrored_signed=mirror[i][SIGNED],original_abs=original[i][ABS],mirrored_abs=mirror[i][ABS],original_fall_rate=original[i]['fall_rate'],mirrored_fall_rate=mirror[i]['fall_rate'],original_prefix_duration_s=original[i]['initial_prefix_mean_duration_s'],mirrored_prefix_duration_s=mirror[i]['initial_prefix_mean_duration_s']) for i,k in enumerate(ids)],
    endpoint_comparison_complete=finite)
   if cmd not in MAIN:
    record['operational_mirror_pattern']='not_applicable_commanded_turn_control'
    record['direction_reversal_inference_applicable']=False
    record['control_scope']='Raw same-command turning performance only. A pure reflection maps positive to negative yaw commands; same-command turn direction is not a zero-yaw bias reversal test. Original reset ordinals are preserved, not silently swapped.'
    comparisons.append(record)
    continue
   record['direction_reversal_inference_applicable']=True
   if finite:
    delta=[m-o for m,o in zip(mirror_signed,original_signed)];signed_sum=[m+o for m,o in zip(mirror_signed,original_signed)]
    ratio=float(np.mean(ma)/np.mean(oa)) if np.mean(oa)>0 else None
    ratio_ci=None
    if len(ids)>1 and ratio is not None:
     draw=np.array(list(itertools.product(range(len(ids)),repeat=len(ids))))
     ratios=np.array(ma)[draw].mean(1)/np.array(oa)[draw].mean(1)
     ratio_ci=list(map(float,np.percentile(ratios,[2.5,97.5])))
    determinate=[abs(o)>1 and abs(m)>1 for o,m in zip(original_signed,mirror_signed)]
    reversed_=[bool(d and o*m<0) for d,o,m in zip(determinate,original_signed,mirror_signed)]
    same=[bool(d and o*m>0) for d,o,m in zip(determinate,original_signed,mirror_signed)]
    comparable=ratio is not None and .8<=ratio<=1.2
    record.update(paired_mirrored_minus_original_signed_degrees=paired_summary(delta),paired_mirrored_plus_original_signed_degrees=paired_summary(signed_sum),
     paired_absolute_difference_degrees=paired_summary([m-o for m,o in zip(ma,oa)]),mean_absolute_mirrored_over_original=ratio,ratio_training_bootstrap95=ratio_ci,
     all_training_seeds_direction_determinate=all(determinate),per_seed_direction_reversed=reversed_,all_training_seeds_direction_reversed=all(reversed_),
     all_training_seeds_same_direction=all(same),absolute_magnitude_comparable_0p8_to1p2=comparable,
     operational_mirror_pattern='reversal_and_comparable' if all(reversed_) and comparable else 'same_direction' if all(same) else 'mixed_or_magnitude_changed')
   else:record['operational_mirror_pattern']='incomplete_or_censored_endpoint_comparison'
   comparisons.append(record)
 default=stats['default:control6000_s1:eval1:stand']
 outputs={'job_statistics.json':stats,'seed_statistics.json':seeds,'group_statistics.json':groups,'mirror_comparisons.json':comparisons,'no_policy_statistics.json':dict(protocol='one1024env20sstand job; no actor called, default PD targets verified every step; no training-seed CI',statistics=default)}
 for name,value in outputs.items():(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
 representative=select_original_representative(cohort)
 baseline_inputs={str(A/n):sha(A/n) for n in ['cohort.json','seed_statistics.json','node_statistics.json']}
 audit=dict(passed=True,formal_jobs=34,formal_command_records=199,initial_trials=203776,per_trial_rows=csv_rows,all_trial_heading_series_audited_commands=full_series,
    maximum_independent_heading_delta_sum_error_radians=max_heading_error,raw_inputs_sha256=raw_inputs,baseline_inputs_sha256=baseline_inputs,
    source_sha256=SOURCE_SHA,zero_training_updates=True,no_new_lineage_baseline_evaluations=True,per_trial_csv_sha256=sha(path),no_censored_seed_silently_dropped=True)
 (ROOT/'analysis_integrity_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 names=[*outputs,'representative_selection.json','analysis_integrity_audit.json','per_trial_heading.csv.gz']
 binding=dict(sources={str(Path(__file__).resolve()):SOURCE_SHA,**{str(REPO/'scripts/locomotion'/n):sha(REPO/'scripts/locomotion'/n) for n in ['issue15_analysis_data.py','issue13_analysis_data.py','issue15_math.py']}},
    outputs={name:sha(ROOT/name) for name in names})
 (ROOT/'analysis_outputs_sha256.json').write_text(json.dumps(binding,indent=2)+'\n')
 print('ISSUE16_ANALYSIS_COMPLETE',recomputed,csv_rows,'representative',representative['model'],representative['eval_seed'],representative['env_id'],flush=True)
if __name__=='__main__':main()
