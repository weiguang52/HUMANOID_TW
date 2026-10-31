"""Apply the prespecified development diagnostic gate, not a policy acceptance gate."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument("--nonregression-only",action="store_true");a=p.parse_args();assert not a.output.exists()
key=lambda r:(r['split'],r['id'],r['mode'])
before={key(r):r for r in json.loads(a.before.read_text())['rows']};after={key(r):r for r in json.loads(a.after.read_text())['rows']};assert before.keys()==after.keys()
checks=[]
for h in ('1','8','32'):
 keys=[k for k,r in before.items() if k[0]=='validation' and r['world_model'][h]['windows']>0]
 assert keys
 nonfinite=[]
 for k in keys:
  b=before[k]['world_model'][h];n=after[k]['world_model'][h];assert b['windows']==n['windows']
  if any('failure' in v or round(v.get('finite_fraction',0)*v['windows'])!=v['windows'] for v in (b,n)):nonfinite.append(k)
 if nonfinite:
  checks.append(dict(horizon=h,metric='finite_rollouts',failed_episodes=nonfinite,passed=False));continue
 metrics=['root_position_error_p95_m','root_orientation_error_p95_rad','root_linear_velocity_rmse_m_s','root_angular_velocity_rmse_rad_s','joint_position_rmse_rad','joint_velocity_rmse_rad_s']
 for metric in metrics:
  old=statistics.median(before[k]['world_model'][h][metric] for k in keys);new=statistics.median(after[k]['world_model'][h][metric] for k in keys);limit=.8 if 'p95' in metric and not a.nonregression_only else 1.05
  checks.append(dict(horizon=h,metric=metric,episodes=len(keys),before_median=old,after_median=new,ratio=new/old if old else None,max_ratio=limit,passed=new<=old*limit))
report=dict(passed=all(c['passed'] for c in checks),scope='Prespecified development gate only; no stable skill or fresh unseen-test claim',checks=checks)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
