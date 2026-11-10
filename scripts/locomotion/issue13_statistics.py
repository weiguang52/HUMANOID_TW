"""Prospectively frozen five-paired-training-seed inference and issue13 A/B rule."""
import numpy as np
INDICES=np.indices((5,)*5).reshape(5,-1).T
def vector(a):
 a=np.asarray(a,dtype=float)
 if a.shape!=(5,) or not np.isfinite(a).all():raise ValueError("Exactly five finite training-seed values required")
 return a
def summary(a):
 a=vector(a)
 return {"per_seed":a.tolist(),"mean":float(a.mean()),"sample_sd":float(a.std(ddof=1)),
  "bootstrap95":np.quantile(a[INDICES].mean(1),[.025,.975]).tolist(),"resamples":3125}
def compare(control,treatment):
 c,t=vector(control),vector(treatment);d=t-c;cb=c[INDICES].mean(1);tb=t[INDICES].mean(1)
 zero=int((cb==0).sum());cm,tm=float(c.mean()),float(t.mean())
 diff=summary(d);ratio=None if cm==0 else tm/cm
 relative_ci=None if zero else np.quantile(tb/cb-1,[.025,.975]).tolist()
 return {"control":summary(c),"treatment":summary(t),"paired_difference":diff,"ratio_of_group_means":ratio,
  "relative_change":None if ratio is None else ratio-1,"relative_change_bootstrap95":relative_ci,
  "zero_control_bootstrap_replicates":zero,"point_not_worse_than5percent":bool(tm<=1.05*cm),
  "point_worse_than5percent_and_positive95":bool(tm>1.05*cm and diff["bootstrap95"][0]>0),
  "multiple_comparisons_corrected":False}
def classify(comparisons):
 if len(comparisons)!=13:raise ValueError("All13frozen clean xy/yaw and combined-outside xy comparisons required")
 if all(x["point_not_worse_than5percent"] for x in comparisons.values()):decision="A_cost_localized_to_recovery_window"
 elif any(x["point_worse_than5percent_and_positive95"] for x in comparisons.values()):decision="B_global_side_effect"
 else:decision="unable_to_distinguish"
 return {"decision":decision,"all13_point_bounds_pass":all(x["point_not_worse_than5percent"] for x in comparisons.values()),
  "B_supporting_comparisons":[k for k,x in comparisons.items() if x["point_worse_than5percent_and_positive95"]],
  "multiple_comparisons_corrected":False,"training_seeds":5,"resamples":3125,
  "issue12_rejection_unchanged":True,"default_promoted":False,"training_updates":0}
