"""Reset-aware startup offset and late heading slope; no rollout changes."""
import numpy as np
def decompose(yaw,start,masks,dt=.02):
    yaw=np.asarray(yaw,dtype=np.float64);start=np.asarray(start,dtype=np.float64)
    steps,n=yaw.shape
    assert steps==1000 and start.shape==(n,) and masks.shape[:2]==(steps,2)
    prefix=np.asarray(masks[:,1,0],dtype=bool)
    assert np.all(~prefix|np.asarray(masks[:,0,0],dtype=bool))
    assert not np.any(np.diff(prefix.astype(np.int8),axis=0)>0),'Initial prefix cannot resume after reset'
    previous=np.concatenate([start[None],yaw[:-1]])
    delta=np.arctan2(np.sin(yaw-previous),np.cos(yaw-previous))
    heading=np.cumsum(np.where(prefix,delta,0.),axis=0)*180/np.pi
    time=(np.arange(steps)+1)*dt
    startup=time<=2.+1e-9; late=(time>=5.-1e-9)&(time<=20.+1e-9)
    startup_mask=prefix[startup]
    startup_complete=startup_mask.all(0)
    startup_prefix=heading[startup][-1]
    startup_signed=np.where(startup_complete,startup_prefix,np.nan)
    weights=prefix[late].astype(float);x=time[late,None];y=heading[late]
    count=weights.sum(0);sx=(weights*x).sum(0);sy=(weights*y).sum(0)
    sxx=(weights*x*x).sum(0);sxy=(weights*x*y).sum(0)
    denom=count*sxx-sx*sx
    slope=np.divide(count*sxy-sx*sy,denom,out=np.full(n,np.nan),where=(count>=2)&(denom>1e-12))
    full=prefix.all(0)
    full_slope=np.where(full,slope,np.nan)
    return dict(startup_signed_degrees=startup_signed,startup_absolute_degrees=np.abs(startup_signed),
                startup_complete=startup_complete,startup_prefix_signed_degrees=startup_prefix,
                startup_prefix_seconds=startup_mask.sum(0)*dt,
                late_full20s_slope_degrees_per_second=full_slope,
                late_initial_prefix_slope_degrees_per_second=slope,
                late_prefix_samples=count.astype(np.int32),survived20s=full,
                initial_prefix_signed_degrees=heading[-1])
def summarize_decomposition(raw):
    result={}
    for key,values in raw.items():
        a=np.asarray(values)
        if a.dtype==bool:
            result[key+'_count']=int(a.sum())
        elif key.endswith('samples'):
            result[key+'_mean']=float(a.mean())
        else:
            finite=a[np.isfinite(a)]
            result[key+'_mean']=float(finite.mean()) if len(finite) else None
            result[key+'_available_trials']=int(len(finite))
    a=raw['late_full20s_slope_degrees_per_second'];a=a[np.isfinite(a)]
    result['late_full20s_absolute_slope_mean']=float(np.abs(a).mean()) if len(a) else None
    return result
