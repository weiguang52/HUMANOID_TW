"""Compare every paired fixed perturbation episode over common actual duration."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
def load(root):
 rows=json.loads((root/'results.json').read_text())['rows'];keys=[(x['id'],x['mode']) for x in rows];assert len(set(keys))==len(keys)
 return dict(zip(keys,rows))
b=load(a.before);c=load(a.after);assert b.keys()==c.keys();rows=[]
for key in b:
 traces=[]
 for root,row in [(a.before,b[key]),(a.after,c[key])]:
  with np.load(root/row['trace']) as z:trace={k:z[k].copy() for k in z.files}
  assert len(trace['state'])==row['steps'];assert all(np.isfinite(v).all() for v in trace.values());assert np.array_equal(trace['next_state'][:-1],trace['state'][1:]);traces.append(trace)
 n=min(len(z['state']) for z in traces);metrics={}
 for label,z in zip(['before','after'],traces):
  s=z['next_state'][:n];q=s[:,3:7];norm=np.linalg.norm(q,axis=1);assert np.all(norm>0);q=q/norm[:,None];metrics[label]=dict(root_tilt_max_deg=float(np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))).max()),root_xy_max_m=float(np.linalg.norm(s[:,:2]-z['state'][0,:2],axis=1).max()))
 rows.append(dict(id=key[0],mode=key[1],common_duration_s=n*.02,before_duration_s=b[key]['steps']*.02,after_duration_s=c[key]['steps']*.02,before_reasons=b[key]['termination_reasons'],after_reasons=c[key]['termination_reasons'],**metrics))
report=dict(scope='Fixed open-loop PD diagnostic only; common actual duration, no extrapolation; forcing seeds are not training seeds',episodes=len(rows),before_full20s=sum(v['steps']==1000 and not v['termination_reasons'] for v in b.values()),after_full20s=sum(v['steps']==1000 and not v['termination_reasons'] for v in c.values()),rows=rows)
for metric in ['root_tilt_max_deg','root_xy_max_m']:report[metric+'_regressions']=[(x['id'],x['mode']) for x in rows if x['after'][metric]>x['before'][metric]]
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
