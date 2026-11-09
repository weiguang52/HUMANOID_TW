"""Training-seed-unit exact empirical bootstrap and illustrative power planning."""
import itertools,math
import numpy as np
from scipy.stats import t,nct,chi2

def exact_indices(n):
    if not 2<=n<=6:raise ValueError("Exact enumeration is intended for2..6 training seeds")
    return np.asarray(list(itertools.product(range(n),repeat=n)),dtype=np.int64)

def paired_seed_summary(requested,applied):
    r=np.asarray(requested,dtype=float);a=np.asarray(applied,dtype=float)
    if r.ndim!=1 or a.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(a).all():raise ValueError("Finite matched vectors required")
    if (r<0).any() or (r>1).any() or (a<0).any() or (a>1).any():raise ValueError("Rates must lie in[0,1]")
    ids=exact_indices(len(r))
    def describe(values):
        means=values[ids].mean(axis=1)
        ci=np.quantile(means,[.025,.975],method='linear')
        ci[np.abs(ci)<1e-12]=0.0  # mathematical zero at finite precision
        return {'rates':values.tolist(),'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'bootstrap95':ci.tolist()}
    difference=a-r;result={'requested':describe(r),'applied':describe(a),'paired_applied_minus_requested':describe(difference),'training_pairs':len(r),'exact_resamples':len(ids)}
    lo,hi=result['paired_applied_minus_requested']['bootstrap95']
    result['effect_label']='supports_lower_applied_rate' if hi<0 else ('supports_higher_applied_rate' if lo>0 else 'cannot_distinguish_at_current_seed_count')
    result['difference_ci_excludes_zero']=hi<0 or lo>0
    return result

def paired_t_power(n,sd,effect=.03,alpha=.05):
    critical=t.ppf(1-alpha/2,n-1);nc=effect*math.sqrt(n)/sd
    return float(nct.sf(critical,n-1,nc)+nct.cdf(-critical,n-1,nc))

def unpaired_t_power(n,sd,effect=.03,alpha=.05):
    df=2*n-2;critical=t.ppf(1-alpha/2,df);nc=effect/(sd*math.sqrt(2/n))
    return float(nct.sf(critical,df,nc)+nct.cdf(-critical,df,nc))

def required_n(sd,effect=.03,alpha=.05,target=.8,paired=True):
    if sd<0 or effect<=0 or not 0<alpha<1 or not 0<target<1:raise ValueError("Invalid planning inputs")
    if sd<=1e-12:return None
    power=paired_t_power if paired else unpaired_t_power
    for n in range(2,10001):
        if power(n,sd,effect,alpha)>=target:return n
    return None

def planning_summary(requested,applied):
    r=np.asarray(requested,dtype=float);a=np.asarray(applied,dtype=float);n=len(r)
    if n<2 or a.shape!=r.shape:raise ValueError("Matched pilot vectors required")
    sd=float((a-r).std(ddof=1));pooled=math.sqrt((r.var(ddof=1)+a.var(ddof=1))/2)
    lower=sd*math.sqrt((n-1)/chi2.ppf(.975,n-1));upper=sd*math.sqrt((n-1)/chi2.ppf(.025,n-1))
    pairs=required_n(sd);per_arm=required_n(pooled,paired=False)
    return {'effect_absolute':.03,'two_sided_alpha':.05,'target_power':.8,'pilot_pairs':n,'paired_difference_sample_sd':sd,'normal_theory_sd95':[lower,upper],'paired_training_pairs_needed':pairs,'additional_pairs_beyond_pilot':None if pairs is None else max(0,pairs-n),'paired_n_using_sd95_bounds':[required_n(lower),required_n(upper)],'unpaired_pooled_sd':pooled,'unpaired_training_seeds_per_arm':per_arm,'assumptions':'noncentral-t planning, normal independent training-pair differences, fixed pilot variance; SD bounds use normal-theory chi-square; conditional on one shared parent; near-boundary rates and n=5 make planning uncertain','degenerate_variance_note':'zero pilot SD does not identify a reliable future sample size' if sd<=1e-12 else None}
