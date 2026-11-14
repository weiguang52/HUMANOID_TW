"""Independently recompute unwrapped heading, OLS, cycle means and seed intervals."""
from pathlib import Path
import json,itertools,numpy as np
from issue20_common import ROOT,sha,save
from issue15_analysis_data import load_yaw
COMMANDS=['forward_03','forward_05','forward_08','stand']
DRAWS=np.indices((5,5,5,5,5)).reshape(5,-1)
def mean(v):
 a=np.asarray(v,dtype=float);a=a[np.isfinite(a)];return float(a.mean()) if len(a) else None
def close(a,b,tol=1e-7):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:assert abs(a-b)<=tol,(a,b,tol)
def check_summary(values,row):
 assert len(row['seeds'])==5
 for x,y in zip(values,row['seeds']):close(x,y)
 if any(x is None for x in values):assert row['mean'] is None and row['ci95'] is None;return
 a=np.asarray(values,float);close(float(a.mean()),row['mean']);expected=np.quantile(a[DRAWS].mean(0),[.025,.975]);assert np.allclose(expected,row['ci95'],atol=1e-8,rtol=0),(expected,row)
def trajectory(y,m,idx,k):
 prefix=np.asarray(m[:,1,0],bool);assert np.all(np.diff(prefix.astype(np.int8),axis=0)<=0);angles=np.vstack([y['heading_start'][None],y['heading_yaw']]).astype(float);unwrapped=np.unwrap(angles,axis=0)-angles[0];delta=np.diff(unwrapped,axis=0)*180/np.pi;heading=unwrapped[1:]*180/np.pi;full=prefix.all(0);startup=np.where(prefix[:100].all(0),heading[99],np.nan);late=np.full(1024,np.nan)
 if full.any():late[full]=np.linalg.lstsq(np.c_[np.arange(250,1001)*.02,np.ones(751)],heading[249:,full],rcond=None)[0][0]
 clock=np.zeros(1024,dtype=np.int64);valid=np.asarray(m[:,0,0],bool)
 for i in range(1000):assert np.array_equal(idx[i],(clock+k)%20);clock=np.where(valid[i],clock+1,0)
 half={};per={};counts=np.zeros(1024,np.int64)
 for side in ['left','right','sum']:
  total=np.zeros(1024,float);pooled_sum=0.;pooled_count=0
  for start in range(0,1000,20):
   good=prefix[start:start+20].all(0);select=(idx[start:start+20]<10) if side=='left' else (idx[start:start+20]>=10) if side=='right' else np.ones((20,1024),bool);value=np.where(select,delta[start:start+20],0).sum(0);total+=np.where(good,value,0);pooled_sum+=value[good].sum();pooled_count+=int(good.sum())
   if side=='left':counts+=good
  half[side]=None if not pooled_count else pooled_sum/pooled_count;per[side]=np.divide(total,counts,out=np.full(1024,np.nan),where=counts>0)
 return {'prefix':prefix,'delta':delta,'full':full,'startup':startup,'late':late,'half':half,'perhalf':per,'counts':counts,'heading':heading}
def main():
 assert (ROOT/'formal_evaluation_complete').exists();records=json.loads((ROOT/'evaluation_records.json').read_text());cache={};jobs={};counts={'formal_jobs':len(records),'command_cells':0,'paired_command_cells':0};max_heading_precision_error=0.
 for row in records:
  assert sha(row['output'])==row['output_sha256'];d=json.loads(Path(row['output']).read_text());folder=Path(d['issue18_tracking']['output']);assert [r['command'] for r in d['results']]==COMMANDS
  analysis=json.loads((ROOT/(row['usd']+'_analysis.json')).read_text())
  for result in d['results']:
   cmd=result['command'];_,y,m,meta=load_yaw(folder,cmd,result);idx=np.load(folder/(cmd+'_phase20.npz'))['pre_action_idx'];t=trajectory(y,m,idx,row['phase']);prefix=t['prefix'];signed=y['heading_delta_sum'][1]*180/np.pi;prec=float(abs(np.where(prefix,t['delta'],0).sum(0)-signed).max());max_heading_precision_error=max(max_heading_precision_error,prec);assert prec<=.006
   metrics={'signed20_degrees':mean(np.where(t['full'],signed,np.nan)),'absolute20_degrees':mean(np.where(t['full'],abs(signed),np.nan)),'startup_signed_degrees_mean':mean(t['startup']),'late_full20s_slope_degrees_per_second_mean':mean(t['late']),'fall_rate':float((y['first_failure_step']>=0).mean()),'initial_prefix_mean_duration_s':float(prefix.sum(0).mean()*.02)}
   for side in ['left','right','sum']:metrics['complete_cycle_'+side+'_mean_degrees']=t['half'][side]
   actual=analysis['job_statistics'][f'{row["training_seed"]}:{row["eval_seed"]}:{row["phase"]}:{cmd}']
   for key,value in metrics.items():close(value,actual[key],1e-7)
   cache[row['usd'],row['training_seed'],row['eval_seed'],row['phase'],cmd]=t;jobs[row['usd'],row['training_seed'],row['eval_seed'],row['phase'],cmd]=metrics;counts['command_cells']+=1
 paired_jobs={}
 for usd in sorted({r['usd'] for r in records}):
  a=json.loads((ROOT/(usd+'_analysis.json')).read_text())
  for ts in range(1,6):
   for es in [1,2,3]:
    for cmd in COMMANDS:
     t0=cache[usd,ts,es,0,cmd];t1=cache[usd,ts,es,10,cmd];full=t0['full']&t1['full'];both=t0['prefix']&t1['prefix'];h0=np.where(t0['prefix'],t0['delta'],0).sum(0);h1=np.where(t1['prefix'],t1['delta'],0).sum(0);c0=np.where(both,t0['delta'],0).sum(0);c1=np.where(both,t1['delta'],0).sum(0);expected={'S20_degrees':np.where(full,h0+h1,np.nan),'D20_degrees':np.where(full,(h0-h1)/2,np.nan),'common_prefix_S_degrees':c0+c1,'common_prefix_D_degrees':(c0-c1)/2,'common_prefix_seconds':both.sum(0)*.02,'half_swap_left_residual_degrees':t1['perhalf']['left']+t0['perhalf']['right'],'half_swap_right_residual_degrees':t1['perhalf']['right']+t0['perhalf']['left']};path=ROOT/'paired_raw'/usd/f's{ts}_eval{es}_{cmd}.npz';z=np.load(path)
     for key,value in expected.items():assert np.allclose(value,z[key],equal_nan=True,atol=1e-7,rtol=0),(path,key)
     assert np.array_equal(z['paired_complete20s'],full);paired_jobs[usd,ts,es,cmd]={k:mean(v) for k,v in expected.items()}|{'paired_complete20s_count':int(full.sum())};counts['paired_command_cells']+=1
  for cmd in COMMANDS:
   for phase in [0,10]:
    for key in next(iter(jobs.values())):
     vals=[None if any(jobs[usd,ts,es,phase,cmd][key] is None for es in [1,2,3]) else float(np.mean([jobs[usd,ts,es,phase,cmd][key] for es in [1,2,3]])) for ts in range(1,6)];check_summary(vals,a['cell_statistics'][f'{phase}:{cmd}'][key])
   for key in a['paired_group_statistics'][cmd]:
    vals=[None if any(paired_jobs[usd,ts,es,cmd][key] is None for es in [1,2,3]) else float(np.mean([paired_jobs[usd,ts,es,cmd][key] for es in [1,2,3]])) for ts in range(1,6)];check_summary(vals,a['paired_group_statistics'][cmd][key])
 save('independent_statistical_audit.json',{'passed':True,**counts,'all_raw_S_D_and_half_swap_arrays_independently_verified':True,'all_primary_cell_metrics_and_seed_intervals_verified':True,'OLS_method':'independent numpy.linalg.lstsq on unwrapped full20s heading','half_cycle_cell_method':'complete eligible cycles pooled as issue19; per-trial paired residuals separately verified','bootstrap':'independent np.indices enumerating all3125 seed resamples','max_float32_accumulator_vs_float64_heading_difference_degrees':max_heading_precision_error});print('ISSUE20_INDEPENDENT_STATISTICAL_AUDIT_PASS',counts,flush=True)
if __name__=='__main__':main()
