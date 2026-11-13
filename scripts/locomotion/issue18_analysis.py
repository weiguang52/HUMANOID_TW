"""Reset-aware scientific calculations for strict projection diagnostic."""
import itertools
import numpy as np
DRAW=np.array(list(itertools.product(range(5),repeat=5)))
def summary(values):
 if len(values)!=5 or any(x is None or not np.isfinite(x) for x in values):return {'mean':None,'ci95':None,'seeds':values,'complete':False}
 a=np.array(values,dtype=float);return {'mean':float(a.mean()),'ci95':np.percentile(a[DRAW].mean(1),[2.5,97.5]).tolist(),'seeds':values,'complete':True}
def complete_mean(values):return None if any(x is None for x in values) else float(np.mean(values))
def half_cycles(yaw,start,masks,phase):
 yaw=np.asarray(yaw,dtype=np.float64);t,n=yaw.shape;assert t==1000 and phase.shape==yaw.shape
 previous=np.concatenate([np.asarray(start,dtype=np.float64)[None],yaw[:-1]])
 delta=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous));valid=masks[:,0,0];prefix=masks[:,1,0]
 assert np.all(~prefix|valid)
 # Reconstruct original episode clock from recorded terminal/reset masks.
 counter=np.zeros(n,dtype=np.int64)
 for k in range(t):
  assert np.array_equal(phase[k],counter),('phase mismatch',k)
  counter=np.where(valid[k],counter+1,0)
 left=phase%20<10; result={};per_trial={}
 for name,scope in [('full_horizon',valid),('initial_prefix',prefix)]:
  for side,partition in [('left',left),('right',~left)]:
   selected=scope&partition;totals=np.where(selected,delta,0).sum(0)*180/np.pi;counts=selected.sum(0)
   per_trial[name+'_'+side+'_yaw_degrees']=totals;per_trial[name+'_'+side+'_steps']=counts
   result[name+'_'+side+'_mean_total_degrees']=float(totals.mean());result[name+'_'+side+'_mean_seconds']=float(counts.mean()*.02)
 # Initial episodes start at index0. Never fill interrupted cycles or reset gaps.
 cycles=delta.reshape(50,20,n);eligible=prefix.reshape(50,20,n).all(1)
 correct=(phase.reshape(50,20,n)%20==np.arange(20)[None,:,None]).all(1);assert np.all(~eligible|correct)
 l=cycles[:,:10].sum(1)*180/np.pi;r=cycles[:,10:].sum(1)*180/np.pi
 for name,values in [('left',l),('right',r),('sum',l+r)]:
  result['complete_cycle_'+name+'_mean_degrees']=float(values[eligible].mean()) if eligible.any() else None
  per_trial['complete_cycle_'+name+'_total_degrees']=np.where(eligible,values,0).sum(0)
 counts=eligible.sum(0);per_trial['complete_cycle_count']=counts
 result.update(complete_cycle_count=int(counts.sum()),initial_prefix_valid_steps=int(prefix.sum()),partial_cycle_valid_steps=int(prefix.sum()-20*counts.sum()),initial_trials=n,full20s_trials=int(prefix.all(0).sum()))
 return result,per_trial
