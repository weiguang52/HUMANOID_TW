"""Paired phase-factor calculations with no joins across resets."""
import numpy as np
from issue19_analysis import decompose

def half_cycles(yaw,start,masks,idx,k):
 yaw=np.asarray(yaw,dtype=np.float64);previous=np.r_[np.asarray(start,dtype=np.float64)[None],yaw[:-1]];delta=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous))*180/np.pi
 prefix=np.asarray(masks[:,1,0],bool);valid=np.asarray(masks[:,0,0],bool);counter=np.zeros(yaw.shape[1],np.int64)
 for step in range(1000):
  assert np.array_equal(idx[step],(counter+k)%20),(step,k)
  counter=np.where(valid[step],counter+1,0)
 # Episode-aligned 20-step complete periods, with side bins determined by actual shifted phase.
 good=prefix.reshape(50,20,-1).all(1);left=idx<10;result={};raw={}
 for side,which in [('left',left),('right',~left),('sum',np.ones_like(left))]:
  cycles=np.where(which,delta,0).reshape(50,20,-1).sum(1);total=np.where(good,cycles,0).sum(0);count=good.sum(0);mean=np.divide(total,count,out=np.full(len(count),np.nan),where=count>0)
  raw['complete_cycle_'+side+'_degrees']=mean;result['complete_cycle_'+side+'_mean_degrees']=float(np.nanmean(mean)) if np.isfinite(mean).any() else None
 raw['complete_cycle_count']=good.sum(0);result['complete_cycle_count']=int(good.sum());return result,raw

def paired(y0,start0,m0,y10,start10,m10):
 a=decompose(y0,start0,m0);b=decompose(y10,start10,m10);full=a['survived20s']&b['survived20s']
 d0=a['initial_prefix_signed_degrees'];d10=b['initial_prefix_signed_degrees'];S=np.where(full,d0+d10,np.nan);D=np.where(full,(d0-d10)/2,np.nan)
 # A separate common-prefix quantity remains defined for censored pairs.
 p0=np.asarray(m0[:,1,0],bool);p10=np.asarray(m10[:,1,0],bool);both=p0&p10
 def prefix(y,s):
  y=np.asarray(y,dtype=np.float64);s=np.asarray(s,dtype=np.float64)
  prev=np.r_[s[None],y[:-1]];inc=np.arctan2(np.sin(y-prev),np.cos(y-prev));return np.where(both,inc,0).sum(0)*180/np.pi
 c0=prefix(y0,start0);c10=prefix(y10,start10)
 return {'S20_degrees':S,'D20_degrees':D,'paired_complete20s':full,'common_prefix_S_degrees':c0+c10,'common_prefix_D_degrees':(c0-c10)/2,'common_prefix_seconds':both.sum(0)*.02}
