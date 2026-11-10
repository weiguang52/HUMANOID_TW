"""Frozen five-training-seed inference for issue12; never bootstrap evaluation envs."""
import numpy as np
from risk_analysis_utils import SEED_INDICES

def vector(x):
    a=np.asarray(x,dtype=float)
    if a.shape!=(5,) or not np.isfinite(a).all():
        raise ValueError("Exactly five finite training-seed values are required")
    return a

def summary(x):
    a=vector(x);boot=a[SEED_INDICES].mean(1)
    return {"per_seed":a.tolist(),"mean":float(a.mean()),"sample_sd":float(a.std(ddof=1)),
            "bootstrap95":np.quantile(boot,[.025,.975]).tolist(),
            "training_seed_sample_size":5,"exact_resamples":3125}

def paired(treatment,control):
    t,c=vector(treatment),vector(control);d=t-c;result=summary(d)
    return {"per_seed_differences":d.tolist(),"mean_difference":result["mean"],
            "bootstrap95":result["bootstrap95"],"sample_sd_of_differences":result["sample_sd"],
            "effective_strictly_negative95":bool(result["bootstrap95"][1]<0),
            "pairing":"same five training seed indices jointly resampled",
            "exact_resamples":3125}

def tracking(treatment,control):
    t,c=vector(treatment),vector(control);tm,cm=float(t.mean()),float(c.mean())
    tb=t[SEED_INDICES].mean(1);cb=c[SEED_INDICES].mean(1)
    zero=int((cb==0).sum())
    ratio=None if cm==0 else tm/cm
    ci=None if zero else np.quantile(tb/cb,[.025,.975]).tolist()
    passed=(tm==0) if cm==0 else tm<=1.10*cm
    return {"control":summary(c),"treatment":summary(t),"paired_difference":paired(t,c),
            "ratio_of_seed_means":ratio,"ratio_bootstrap95":ci,
            "zero_control_bootstrap_replicates":zero,
            "not_worse_than_10percent":bool(passed)}

def paired_delay_rr_change(treatment20,treatment0,control20,control0):
    values=[]
    for x in [treatment20,treatment0,control20,control0]:
        a=np.asarray(x,dtype=float)
        if a.shape!=(5,2) or np.any(a[:,1]<=0):
            return {"estimable":False,"reason":"a seed lacks target/reference exposure"}
        values.append(a[:,0]/a[:,1])
    t20,t0,c20,c0=values
    t20b,t0b,c20b,c0b=[v[SEED_INDICES].mean(1) for v in values]
    bad=(t0b==0)|(c0b==0)
    point=None if t0.mean()==0 or c0.mean()==0 else float(t20.mean()/t0.mean()-c20.mean()/c0.mean())
    ci=None if bad.any() else np.quantile(t20b/t0b-c20b/c0b,[.025,.975]).tolist()
    return {"estimable":point is not None,"treatment_minus_control_rr":point,
            "paired_seed_bootstrap95":ci,"zero_reference_bootstrap_replicates":int(bad.sum()),
            "no_pseudocount_or_infinite_rr":True,"exact_resamples":3125}

def decision(low_control,low_treatment,mid_control,mid_treatment,xy):
    effect=paired(low_treatment,low_control);low=summary(low_treatment)
    flags={"low_speed_effect_strictly_negative95":effect["effective_strictly_negative95"],
           "treatment_low_speed_mean_ci_upper_below_1percent":bool(low["bootstrap95"][1]<.01),
           "mid_speed_treatment_mean_not_above_control":bool(vector(mid_treatment).mean()<=vector(mid_control).mean()),
           "xy_tracking_at_every_command_not_worse_than_10percent":all(v["not_worse_than_10percent"] for v in xy.values())}
    all_pass=all(flags.values())
    if not flags["low_speed_effect_strictly_negative95"] or not flags["xy_tracking_at_every_command_not_worse_than_10percent"]:
        action="stop_this_scheme_no_extra_budget_or_weight_scan"
    elif not all_pass:
        action="effect_detected_but_not_all_frozen_targets_met_no_default_promotion"
    else:
        action="all_frozen_primary_and_tracking_targets_met_opt_in_retained"
    return {"frozen_gates":flags,"all_primary_tracking_gates_passed":all_pass,
            "low_speed_paired_effect":effect,"treatment_low_speed":low,"decision":action,
            "default_promoted":False,"budget_beyond_frozen_1500_per_run_authorized_or_used":False,
            "weight_scan":False,"secondary_results_are_descriptive_not_added_posthoc_gates":True}


def _integer_vector(counts,n):
    a=vector(counts)
    if n<=0 or np.any(a<0) or np.any(a>n) or not np.equal(a,np.floor(a)).all():
        raise ValueError('Five valid integer fall counts and positive common denominator required')
    return a.astype(np.int64)


def _integer_lattice_summary(a,n):
    sums=np.sort(a[SEED_INDICES].sum(1))
    # (3125-1)*.025=78.1; (3125-1)*.975=3045.9, exactly.
    # Preserve the linear-percentile numerator as integer tenths of count sums.
    nums=[int(9*sums[78]+sums[79]),int(sums[3045]+9*sums[3046])]
    denominator=50*n
    return {'per_seed':(a/n).tolist(),'mean':int(a.sum())/(5*n),
            'sample_sd':float((a/n).std(ddof=1)),
            'bootstrap95':[x/denominator for x in nums],
            'bootstrap95_integer_numerators':nums,'bootstrap95_integer_denominator':denominator,
            'training_seed_sample_size':5,'exact_resamples':3125}


def fall_summary(counts,n=3072):
    return _integer_lattice_summary(_integer_vector(counts,n),n)


def paired_falls(treatment_counts,control_counts,n=3072):
    t=_integer_vector(treatment_counts,n);c=_integer_vector(control_counts,n)
    d=t-c;s=_integer_lattice_summary(d,n)
    return {'per_seed_differences':(d/n).tolist(),'mean_difference':s['mean'],
            'bootstrap95':s['bootstrap95'],'sample_sd_of_differences':s['sample_sd'],
            'bootstrap95_integer_numerators':s['bootstrap95_integer_numerators'],
            'bootstrap95_integer_denominator':s['bootstrap95_integer_denominator'],
            'effective_strictly_negative95':s['bootstrap95_integer_numerators'][1]<0,
            'pairing':'same five training seed indices jointly resampled; exact count lattice',
            'exact_resamples':3125}


def decision_counts(low_control,low_treatment,mid_control,mid_treatment,xy,n=3072):
    effect=paired_falls(low_treatment,low_control,n);low=fall_summary(low_treatment,n)
    flags={'low_speed_effect_strictly_negative95':effect['effective_strictly_negative95'],
           'treatment_low_speed_mean_ci_upper_below_1percent':low['bootstrap95_integer_numerators'][1]*100<low['bootstrap95_integer_denominator'],
           'mid_speed_treatment_mean_not_above_control':int(_integer_vector(mid_treatment,n).sum())<=int(_integer_vector(mid_control,n).sum()),
           'xy_tracking_at_every_command_not_worse_than_10percent':all(x['not_worse_than_10percent'] for x in xy.values())}
    all_pass=all(flags.values())
    if not flags['low_speed_effect_strictly_negative95'] or not flags['xy_tracking_at_every_command_not_worse_than_10percent']:
        action='stop_this_scheme_no_extra_budget_or_weight_scan'
    elif not all_pass:action='effect_detected_but_not_all_frozen_targets_met_no_default_promotion'
    else:action='all_frozen_primary_and_tracking_targets_met_opt_in_retained'
    return {'frozen_gates':flags,'all_primary_tracking_gates_passed':all_pass,
            'low_speed_paired_effect':effect,'treatment_low_speed':low,'decision':action,
            'default_promoted':False,'budget_beyond_frozen_1500_per_run_authorized_or_used':False,
            'weight_scan':False,'secondary_results_are_descriptive_not_added_posthoc_gates':True,
            'strict_fall_gates_use_exact_integer_percentile_numerators':True}
