"""Outcome-blind feature bins and fixed-window risk units for issue11."""
import itertools,math
import numpy as np
from confidence_stats import wilson
AMP_CUTS=np.array([.3*math.sqrt(x/math.pi) for x in [1,2,3]])
MASS_CUTS=np.array([.75,1.,1.25])
SEED_INDICES=np.asarray(list(itertools.product(range(5),repeat=5)))

def quartile(x,cuts):return np.searchsorted(np.asarray(cuts),np.asarray(x),side='right')+1

def directions(delta):
 a=np.asarray(delta);longitudinal=np.abs(a[:,0])>=np.abs(a[:,1])
 return np.where(longitudinal,np.where(a[:,0]>=0,'forward','back'),'side')

def phase_bin(phase,support):
 p=np.asarray(phase);s=np.asarray(support);out=np.full(len(s),-1.)
 out[s==1]=p[s==1,0];out[s==2]=p[s==2,1]
 both=(s==3)&(p>=0).all(1);out[both]=p[both].mean(1)
 return np.where(out<0,0,np.minimum(3,np.floor(np.maximum(out,0)*4)).astype(int)+1)

def push_windows(env,step,at_risk,first_failure,horizon_steps=1000,window_steps=50):
 env=np.asarray(env,dtype=int);step=np.asarray(step,dtype=int);at_risk=np.asarray(at_risk,dtype=bool);failure=np.asarray(first_failure,dtype=int)[env]
 event=(failure>step)&(failure<=step+window_steps)
 next_step=np.full(len(step),horizon_steps+window_steps+1,dtype=int)
 order=np.lexsort((step,env));same=env[order[:-1]]==env[order[1:]];next_step[order[:-1][same]]=step[order[1:][same]]
 overlap=next_step<step+window_steps
 event &= failure<=next_step
 admin=step+window_steps<=horizon_steps
 censored=at_risk&admin&overlap&~event
 eligible=at_risk&admin&(~overlap|event)
 return eligible,event&eligible,censored

def risk(k,n):
 k=int(k);n=int(n)
 if n<0 or k<0 or k>n:raise ValueError('invalid binomial counts')
 if n==0:return {'falls':k,'n':n,'rate':None,'wilson95_low':None,'wilson95_high':None}
 lo,hi=wilson(k,n)
 if k==0:lo=0.
 if k==n:hi=1.
 return {'falls':k,'n':n,'rate':k/n,'wilson95_low':lo,'wilson95_high':hi}

def seed_effect(counts,reference):
 a=np.asarray(counts,dtype=float);b=np.asarray(reference,dtype=float)
 if a.shape!=(5,2) or b.shape!=(5,2) or np.any(a[:,1]<=0) or np.any(b[:,1]<=0):return {'estimable':False,'reason':'not all five seeds have exposure in target/reference'}
 ar=a[:,0]/a[:,1];br=b[:,0]/b[:,1];am=ar.mean();bm=br.mean()
 ab=ar[SEED_INDICES].mean(1);bb=br[SEED_INDICES].mean(1);diff=ab-bb
 rr=None if bm==0 else am/bm;zero=int((bb==0).sum());rrci=None if zero else np.quantile(ab/bb,[.025,.975]).tolist()
 return {'estimable':True,'target_seed_rates':ar.tolist(),'reference_seed_rates':br.tolist(),'target_mean':float(am),'reference_mean':float(bm),'target_sample_sd':float(ar.std(ddof=1)),'reference_sample_sd':float(br.std(ddof=1)),'risk_difference':float(am-bm),'difference_bootstrap95':np.quantile(diff,[.025,.975]).tolist(),'relative_risk':None if rr is None else float(rr),'relative_risk_bootstrap95':rrci,'zero_reference_bootstrap_replicates':zero,'training_seeds':5,'exact_resamples':3125}
