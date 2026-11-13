"""Frozen five-seed acceptance; missing endpoints never silently drop a seed."""
import itertools,json,hashlib
from pathlib import Path
import numpy as np
from issue15_math import paired_summary
ROOT=Path('/root/gpufree-data/tw_stage1/issue17_20261113');MAIN=['stand','forward_03','forward_05','forward_08'];ABS='full20s_survivor_heading_mean_abs_degrees';SIGNED='full20s_survivor_heading_signed_mean_degrees'
def complete(v):return None if any(x is None for x in v) else float(np.mean(v))
def summary(v):return None if any(x is None for x in v) else paired_summary(v)
def diff(a,b):return None if any(x is None for x in a+b) else paired_summary([x-y for x,y in zip(a,b)])
def upper_negative(s):return bool(s is not None and s['bootstrap95'] is not None and s['bootstrap95'][1]<0)
def main():
 assert json.loads((ROOT/'raw_analysis_audit.json').read_text())['passed'];raw=json.loads((ROOT/'job_statistics.json').read_text());seeds={};groups={}
 for group in ['control_int','sym_int']:
  for protocol,commands in [('clean',['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']),('mirror',['forward_03','yaw_positive','yaw_negative','forward_05','forward_08','stand']),('combined',['forward_03','forward_05','forward_08'])]:
   for cmd in commands:
    key=f'{protocol}:{group}:{cmd}';by={}
    for seed in range(1,6):
     values=[raw[f'{protocol}:{group}_s{seed}:eval{es}:{cmd}'] for es in [1,2,3]]
     by[str(seed)]={k:complete([v[k] for v in values]) for k in values[0]}
     by[str(seed)]['full20s_survivors_total_3evals']=sum(v['full20s_heading_survivors'] for v in values);by[str(seed)]['initial_trials_total_3evals']=3072
    seeds[key]=by;groups[key]={k:summary([v[k] for v in by.values()]) for k in next(iter(by.values()))}
 def vals(protocol,group,cmd,metric):return [seeds[f'{protocol}:{group}:{cmd}'][str(i)][metric] for i in range(1,6)]
 comparisons={};gates={}
 primary=diff(vals('clean','sym_int','forward_05',ABS),vals('clean','control_int','forward_05',ABS));comparisons['primary']=primary;gates['primary']=upper_negative(primary)
 for cmd in MAIN:
  endpoint=summary(vals('clean','sym_int',cmd,ABS));gates['heading_'+cmd]=bool(endpoint is not None and endpoint['mean']<=30)
  sums={}
  for group in ['control_int','sym_int']:
   o=vals('clean',group,cmd,SIGNED);m=vals('mirror',group,cmd,SIGNED);sums[group]=[None if a is None or b is None else abs(a+b) for a,b in zip(o,m)]
  v=diff(sums['sym_int'],sums['control_int']);comparisons['abs_O_plus_M_'+cmd]=dict(per_seed=sums,paired_difference=v);gates['mirror_'+cmd]=upper_negative(v)
 draws=np.array(list(itertools.product(range(5),repeat=5)))
 for cmd,metric,bound in [('forward_05','full_xy_error',.05),('forward_08','full_xy_error',.05),('yaw_positive','full_yaw_error',.10),('yaw_negative','full_yaw_error',.10)]:
  a=vals('clean','sym_int',cmd,metric);b=vals('clean','control_int',cmd,metric);ratio=None;interval=None
  if all(v is not None for v in a+b) and np.mean(b)>0:
   ratio=float(np.mean(a)/np.mean(b)-1);bs=np.array(b)[draws].mean(1);assert (bs>0).all();samples=np.array(a)[draws].mean(1)/bs-1;interval=np.percentile(samples,[2.5,97.5]).tolist()
  comparisons['tracking_'+cmd]=dict(relative_change=ratio,paired_ratio_bootstrap95=interval,paired_difference=diff(a,b),bound=bound);gates['tracking_'+cmd]=bool(ratio is not None and ratio<=bound)
 f03=summary(vals('combined','sym_int','forward_03','fall_rate'));f05new=vals('combined','sym_int','forward_05','fall_rate');f05old=vals('combined','control_int','forward_05','fall_rate')
 gates['combined_f03']=bool(f03 is not None and f03['bootstrap95'][1]<.01);gates['combined_f05']=bool(complete(f05new) is not None and complete(f05new)<=complete(f05old))
 comparisons['combined_f03_fall']=f03;comparisons['combined_f05_fall']=dict(new=summary(f05new),control=summary(f05old),paired_difference=diff(f05new,f05old))
 equiv=json.loads((ROOT/'equivariance_statistics.json').read_text());assert equiv['count']==102400
 e={g:[next(r['mean'] for r in equiv['results'] if r['group']==g and r['training_seed']==i) for i in range(1,6)] for g in ['control_int','sym_int','control6000']};ed=diff(e['sym_int'],e['control_int']);comparisons['action_E']=dict(per_seed=e,paired_difference=ed);gates['action_E']=bool(ed is not None and ed['mean']<0)
 old=json.loads(Path('validation_artifacts/stage1_issue15_20261111/seed_statistics.json').read_text());secondary={}
 for cmd in MAIN:
  secondary[cmd]={metric:diff(vals('clean','control_int',cmd,metric),[old['control6000'][cmd][str(i)][metric] for i in range(1,6)]) for metric in [ABS,SIGNED,'full_xy_error','full_yaw_error','full_vx_mean']}
 for name,value in [('seed_statistics.json',seeds),('group_statistics.json',groups),('paired_comparisons.json',comparisons),('old_control_secondary.json',secondary),('acceptance.json',dict(primary_pass=gates['primary'],all_gates_pass=all(gates.values()),gates=gates,training_seed_units=5,intervals='paired unadjusted bootstrap95; no censored seed dropped',next_action_if_primary_fail='Stop hypothesis, no added budget/weight scan/phase change; future symmetric USD2x2 diagnostic only'))]:
  (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
 print('ISSUE17_ACCEPTANCE_COMPUTED',gates['primary'],all(gates.values()),flush=True)
if __name__=='__main__':main()
