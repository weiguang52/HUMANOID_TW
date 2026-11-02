"""Measure native global-yaw covariance; no training or policy acceptance."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queries',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.queries/'query_plan.json').read_text());audit=json.loads((a.collection/'independent_audit.json').read_text());assert audit['passed']
assert hashlib.sha256((a.collection/'transitions.npz').read_bytes()).hexdigest()==audit['transitions_sha256']
with np.load(a.queries/'queries.npz') as z:q={k:z[k] for k in z.files}
with np.load(a.collection/'transitions.npz') as z:d={k:z[k] for k in z.files}
def rotate(s,angle):
 r=Rotation.from_euler('z',angle,degrees=True);out=s.astype(np.float64).copy()
 for k in [0,7,10]:out[k:k+3]=r.apply(s[k:k+3])
 quat=(r*Rotation.from_quat(s[[4,5,6,3]])).as_quat();out[3:7]=quat[[3,0,1,2]];return out
# Independent SciPy transforms audit the requested state and native body/world convention.
for row in plan['rows']:
 i=row['query'];angle=row['yaw_degrees'];expected=rotate(q['source_state'][i],angle)
 assert np.max(np.abs(expected-q['query_state'][i]))<2e-6
 r=Rotation.from_euler('z',angle,degrees=True)
 for key in ['qvel','warmstart']:
  x=q['source_native_'+key][i].copy();x[:3]=r.apply(x[:3]);np.testing.assert_allclose(x,q['query_native_'+key][i],rtol=0,atol=1e-6)
 assert np.array_equal(q['query_native_qpos'][i,7:],q['source_native_qpos'][i,7:])
lookup={(int(i),bool(w),int(rep)):j for j,(i,w,rep) in enumerate(zip(d['query'],d['warm_source'],d['repeat']))};groups={};records=[]
for row in plan['rows']:
 i=row['query']
 if (i,True,0) not in lookup:continue
 zero=next(x['query'] for x in plan['rows'] if x['anchor_query']==row['anchor_query'] and x['yaw_degrees']==0)
 for warm in [True,False]:
  pred=d['next_state'][lookup[i,warm,0]].astype(float);target=rotate(d['next_state'][lookup[zero,warm,0]],row['yaw_degrees'])
  pred[3:7]/=np.linalg.norm(pred[3:7]);target[3:7]*=1 if np.dot(pred[3:7],target[3:7])>=0 else -1
  err=np.abs(pred-target);rec=dict(query=i,anchor=row['anchor_query'],angle=row['yaw_degrees'],warm_source=warm,max_component=float(err.max()),root_position_m=float(np.linalg.norm(err[:3])),root_linear_velocity=float(np.linalg.norm(err[7:10])),root_angular_velocity=float(np.linalg.norm(err[10:13])),joint_velocity=float(np.linalg.norm(err[43:73])),terminated=bool(d['terminated'][lookup[i,warm,0]]))
  records.append(rec);groups.setdefault((row['yaw_degrees'],warm),[]).append(rec)
summary=[]
for (angle,warm),rs in groups.items():
 summary.append(dict(angle=angle,warm_source=warm,n=len(rs),failed_1e_5=sum(x['max_component']>1e-5 for x in rs),max_component=max(x['max_component'] for x in rs),root_position_max_m=max(x['root_position_m'] for x in rs)))
result=dict(summary=summary,records=records,independent_requested_transform_passed=True,training_approved=False,promotion_approved=False,scope='Physical covariance only; original command is not rotated, termination flags are retained but not a covariance gate. A failure does not establish friction-cone causality.')
(a.collection/'yaw_covariance_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary))
