"""Fit-only comparison of old derivative anchors with current physical traces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ['branches','collection','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
branch_plan=json.loads((a.branches/'plan.json').read_text())
integrity=json.loads((a.branches.parent/'serial_integrity.json').read_text());assert integrity['passed']
hashes={x['file']:x['sha256'] for x in integrity['files']}
old={x['trace']:x for x in branch_plan['source_files']}
current={x['trace']:x for x in json.loads((a.collection/'results.json').read_text())['rows']};assert len(current)==32
left=json.loads((a.branches/'control_mapping.json').read_text());right=json.loads((a.collection/'control_mapping.json').read_text())
for key in ['state_joint_names','action_joint_names']:assert left[key]==right[key]
rows=[];unavailable=[];loaded={}
for row in json.loads((a.branches/'results.json').read_text())['rows']:
 if row['split']!='fit' or row['level']!=0:continue
 trace=row['source_trace'];source=current[trace];assert source['split']=='fit' and source['source_family']==row['source_family']
 path=a.branches/row['file'];assert path.resolve().is_relative_to(a.branches.resolve())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[row['file']]==row['sha256']
 if trace not in loaded:
  q=a.collection/trace;assert hashlib.sha256(q.read_bytes()).hexdigest()==source['sha256']
  with np.load(q,allow_pickle=False) as z:loaded[trace]={k:z[k].copy() for k in ['state','action','context']}
 data=loaded[trace];index=row['index']
 if index>=len(data['state']):unavailable.append(dict(trace=trace,index=index,current_steps=len(data['state'])));continue
 with np.load(path,allow_pickle=False) as z:
  before=z['state'][0].astype(float);action=z['action'][0].astype(float);context=z['context'][0].astype(float)
  assert np.max(np.abs(z['state']-z['state'][0]))<=2e-6
 after=data['state'][index].astype(float);delta=before-after
 dot=abs(float(before[3:7]@after[3:7]))/(np.linalg.norm(before[3:7])*np.linalg.norm(after[3:7]))
 rows.append(dict(trace=trace,index=index,reference_sha_changed=old[trace]['reference_sha256']!=source['reference_sha256'],root_position_error_m=float(np.linalg.norm(delta[:3])),root_orientation_error_rad=float(2*np.arccos(np.clip(dot,0,1))),joint_position_rmse_rad=float(np.sqrt(np.mean(delta[13:43]**2))),root_linear_velocity_error_m_s=float(np.linalg.norm(delta[7:10])),root_angular_velocity_error_rad_s=float(np.linalg.norm(delta[10:13])),joint_velocity_rmse_rad_s=float(np.sqrt(np.mean(delta[43:73]**2))),action_rmse=float(np.sqrt(np.mean((action-data['action'][index])**2))),context_max_error=float(np.max(np.abs(context-data['context'][index])))))
assert len(rows)+len(unavailable)==48 and len(loaded)==16
metrics=[k for k,v in rows[0].items() if isinstance(v,float)]
summary={k:dict(median=float(np.median([x[k] for x in rows])),maximum=max(x[k] for x in rows)) for k in metrics}
report=dict(fit_anchors_total=48,matched=len(rows),unavailable=unavailable,fit_traces=16,validation_arrays_loaded=False,branch_plan_sha256=hashlib.sha256((a.branches/'plan.json').read_bytes()).hexdigest(),current_results_sha256=hashlib.sha256((a.collection/'results.json').read_bytes()).hexdigest(),summary=summary,rows=rows,scope='Physical state/action coverage comparison, not label invalidation or causal proof. No fitting or acceptance gate change.')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
