"""Prospectively specified issue15 scalar and spectral definitions."""
import itertools
import numpy as np
def bias_variance(count,sums,squares,target=0.):
 c=np.asarray(count,dtype=float);s=np.asarray(sums,dtype=float);q=np.asarray(squares,dtype=float)
 mean=np.divide(s,c,out=np.full_like(s,np.nan),where=c>0)
 var=np.maximum(np.divide(q,c,out=np.full_like(q,np.nan),where=c>0)-mean**2,0)
 total=float(np.sum(q-2*target*s+target*target*c))
 within=float(np.nansum(c*var));bias=float(np.nansum(c*(mean-target)**2))
 assert abs(within+bias-total)<1e-7*max(1.,abs(total))
 return {"mean_signed_bias":float(np.sum(s-target*c)/np.sum(c)),"mean_abs_trial_bias":float(np.nanmean(abs(mean-target))),
 "rms_trial_bias":float(np.sqrt(np.nansum(c*(mean-target)**2)/np.sum(c))),
 "mean_trial_temporal_sd":float(np.nanmean(np.sqrt(var))),
 "within_energy_fraction":None if total<=1e-20 else within/total,"within_energy":within,"bias_energy":bias}
def paired_summary(values):
 x=np.asarray(values,dtype=float);assert x.ndim==1 and np.isfinite(x).all()
 n=len(x);mean=float(x.mean())
 if n==1:return {"mean":mean,"training_seeds":1,"bootstrap95":None,"sd":None}
 idx=np.asarray(list(itertools.product(range(n),repeat=n)))
 ci=np.percentile(x[idx].mean(1),[2.5,97.5])
 return {"mean":mean,"training_seeds":n,"bootstrap95":ci.tolist(),"sd":float(x.std(ddof=1))}
def spectrum(x,dt=.02,lo=.2,hi=10.):
 x=np.asarray(x,dtype=float);assert x.ndim==1 and len(x)>=250
 y=(x-x.mean())*np.hanning(len(x));p=abs(np.fft.rfft(y))**2
 f=np.fft.rfftfreq(len(x),dt);chosen=(f>=lo)&(f<=hi)
 if float(p[chosen].sum())<=1e-20:return None
 j=np.flatnonzero(chosen)[np.argmax(p[chosen])]
 return {"peak_hz":float(f[j]),"resolution_hz":1/(len(x)*dt),"frequencies":f,"power":p,"samples":len(x)}
def contact_frequencies(contact,dt=.02):
 a=np.asarray(contact,dtype=bool);assert a.ndim==2 and a.shape[1]==2
 counts=[]
 for i in range(2):
  rises=np.flatnonzero(a[1:,i]&~a[:-1,i])+1
  accepted=[];last=-10**9
  for t in rises:
   if t-last>=round(.1/dt):accepted.append(t);last=t
  counts.append(len(accepted))
 duration=(len(a)-1)*dt
 return {"left_stride_hz":counts[0]/duration,"right_stride_hz":counts[1]/duration,"step_hz":sum(counts)/duration,"touchdowns":counts,"debounce_s":.1}
