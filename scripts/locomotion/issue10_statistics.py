"""Frozen issue10 gate; training seeds, not trials, are the units."""
import itertools
import numpy as np
TOL=1e-12

def validate(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(5,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
        raise ValueError('exactly five finite seed rates in [0,1] required')
    return x

def describe(values):
    x=validate(values);indices=np.asarray(list(itertools.product(range(5),repeat=5)))
    bounds=np.quantile(x[indices].mean(1),[.025,.975])
    return {'rates':x.tolist(),'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),'bootstrap95':bounds.tolist(),'training_seeds':5,'exact_resamples':3125}

def paired_decline(before,after):
    x=validate(before)-validate(after);indices=np.asarray(list(itertools.product(range(5),repeat=5)))
    bounds=np.quantile(x[indices].mean(1),[.025,.975]);bounds[np.abs(bounds)<TOL]=0
    return {'rates':x.tolist(),'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),'bootstrap95':bounds.tolist(),'excludes_zero':bool(bounds[0]>TOL or bounds[1]<-TOL)}

def gate(before,after):
    b=describe(before);a=describe(after);d=paired_decline(before,after)
    threshold=.5*a['sample_sd'];passed=d['mean']>TOL and d['mean']>threshold+TOL
    return {'primary_speed':'forward_05','mean1000':b['mean'],'mean1500':a['mean'],'decline':d['mean'],'sd1500':a['sample_sd'],'half_sd1500':threshold,'sd1000_context':b['sample_sd'],'paired_difference_sd_context':d['sample_sd'],'gate_passed':bool(passed),'decision':'still_improving_enter_B' if passed else 'converged_under_frozen_rule_stop_without_B','numerical_tolerance':TOL,'sd_timepoint_interpretation':'prospectively frozen endpoint1500 sampleSD(ddof1)','not_asymptotic_convergence_proof':True}
