"""Prepare fixed fit root-pose probes from all archived native anchor queries."""
import argparse
from pathlib import Path
import json,hashlib,math,datetime
import numpy as np
arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('--source-queries',type=Path,required=True);arg.add_argument('--output',type=Path,required=True);arg.add_argument('--full-axes-levels',action='store_true');a=arg.parse_args();p=a.output;source=a.source_queries;assert not p.exists();p.mkdir(parents=True);old=json.loads((source/'query_plan.json').read_text());f=source/'queries.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==old['query_sha256']
with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
anchors=[row for row in old['rows'] if row['depth']==0];assert len(anchors)==48
records={k:[] for k in d};rows=[]
def multiply(a,b):
 w,x,y,z=a;v,i,j,k=b
 return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])
cases=[(-1,0,0)]+[(axis,sign,level) for level in [1,2] for axis in range(6) for sign in [1,-1]] if a.full_axes_levels else [(-1,0,0),(2,1,1),(2,-1,1),(3,1,1),(3,-1,1),(4,1,1),(4,-1,1)]
for anchor in anchors:
 for axis,sign,level in cases:
  i=anchor['query'];state=d['source_state'][i].copy();delta=0. if axis<0 else (.0005 if axis<3 else .001)*level
  if axis in [0,1,2]:state[axis]+=sign*delta
  elif axis in [3,4,5]:
   q=state[3:7].astype(np.float64);q/=np.linalg.norm(q);rotation=np.zeros(4);rotation[0]=math.cos(sign*delta/2);rotation[axis-2]=math.sin(sign*delta/2);state[3:7]=multiply(rotation,q).astype(np.float32)
  assert np.isfinite(state).all() and abs(np.linalg.norm(state[3:7])-1)<1e-5
  if axis<0:assert np.array_equal(state,d['source_state'][i])
  assert np.array_equal(state[7:],d['source_state'][i,7:])
  for key in records:records[key].append(state if key=='query_state' else d[key][i])
  row={key:anchor[key] for key in ['trace','id','source_family','start','index','reference_frame','split']};row.update(query=len(rows),anchor_query=i,is_native_anchor=axis<0,perturb_axis=axis,sign=sign,delta=delta,has_archived_next_label=axis<0)
  if a.full_axes_levels:row['level']=level
  rows.append(row)
assert len(rows)==(1200 if a.full_axes_levels else 336) and len({x['source_family'] for x in rows})==4
f=p/'queries.npz';assert not f.exists();np.savez_compressed(f,**{k:np.stack(v) for k,v in records.items()})
plan=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),contract='fit_dynamic_root_pose_full_v1' if a.full_axes_levels else 'fit_dynamic_root_pose_probe_v1',queries=len(rows),rows=rows,sources=old['sources'],collection=old['collection'],manifest=old['manifest'],query_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source_query_plan_sha256=hashlib.sha256((source/'query_plan.json').read_bytes()).hexdigest(),validation_arrays_loaded=False,optimizer_updates=0,training_approved=False,promotion_approved=False,scope='All48 existing native fit anchors, zero plus z/roll/pitch +/-existing base .0005m/.001rad. World velocities, joints/action/context fixed; rotations left-multiply world frame. No filtering; source labels valid only atzero. Conditional warmstart/repeat diagnosis, no policy acceptance.')
if a.full_axes_levels:
 plan.update(axes=list(range(6)),levels=[1,2],base_delta=[.0005]*3+[.001]*3,scope='All48 fixed fit anchors; zero plus all6 world-pose axes xlevels1/2 xsigns. Same base scales as static root-pose response data. World velocities/joints/action/context fixed, world rotations left-multiplied. No filtering or policy acceptance.')
(p/'query_plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(dict(queries=len(rows),native_anchors=48,pose_queries=len(rows)-48,sha256=plan['query_sha256'])))
