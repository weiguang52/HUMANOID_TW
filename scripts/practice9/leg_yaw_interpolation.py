"""Port of upstream interpolate_leg_yaw: preserve orthants during resampling."""
import numpy as np


def interpolate(values, source_t, target_t, pairs):
    values=np.asarray(values,dtype=float)
    result=np.column_stack([np.interp(target_t,source_t,c) for c in values.T])
    left=np.clip(np.searchsorted(source_t,target_t,side='right')-1,0,len(values)-2)
    ratio=np.clip((target_t-source_t[left])/(source_t[left+1]-source_t[left]),0,1)
    a,b=values[left],values[left+1]
    for hip,ankle in pairs:
        if np.any(values[:,hip]*values[:,ankle]<-1.e-8):
            raise ValueError('Input violates upstream same-direction leg yaw contract')
        switch=(a[:,hip]+a[:,ankle])*(b[:,hip]+b[:,ankle])<0
        scale=np.abs(1-2*ratio)
        for joint in (hip,ankle):
            bridged=scale*np.where(ratio<=.5,a[:,joint],b[:,joint])
            result[switch,joint]=bridged[switch]
    return result
