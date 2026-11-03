"""Fixed fit-only global-yaw probes with explicit native-coordinate transforms."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-queries',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
old=json.loads((a.source_queries/'query_plan.json').read_text());f=a.source_queries/'queries.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==old['query_sha256']
with np.load(f,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
expanded=old.get('contract')=='expanded_fit_native_anchor_v1'
anchors=[x for x in old['rows'] if x['depth']==0];assert len(anchors)==(288 if expanded else 48) and len({x['source_family'] for x in anchors})==(24 if expanded else 4)
angles=[0,30,45,90,180]
def rotate_vector(v,deg):
 out=v.copy()
 if deg==0:return out
 if deg==90:out[0],out[1]=-v[1],v[0];return out
 if deg==180:out[:2]=-v[:2];return out
 theta=math.radians(deg);c,s=math.cos(theta),math.sin(theta);out[0]=c*float(v[0])-s*float(v[1]);out[1]=s*float(v[0])+c*float(v[1]);return out

def rotate_quaternion(q,deg):
 if deg==0:return q.copy()
 theta=math.radians(deg)/2;c,s=math.cos(theta),math.sin(theta);w,x,y,z=q.astype(float)
 return np.array([c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w],dtype=q.dtype)

def rotate_state(state,deg):
 out=state.copy()
 for offset in [0,7,10]:out[offset:offset+3]=rotate_vector(state[offset:offset+3],deg)
 out[3:7]=rotate_quaternion(state[3:7],deg);return out

records={k:[] for k in source};records.update(query_native_qpos=[],query_native_qvel=[],query_native_warmstart=[]);rows=[]
for anchor in anchors:
 i=anchor['query']
 for angle in angles:
  state=rotate_state(source['source_state'][i],angle);qpos=source['source_native_qpos'][i].copy();qvel=source['source_native_qvel'][i].copy();warm=source['source_native_warmstart'][i].copy()
  qpos[:3]=rotate_vector(qpos[:3],angle);qpos[3:7]=rotate_quaternion(qpos[3:7],angle)
  qvel[:3]=rotate_vector(qvel[:3],angle);warm[:3]=rotate_vector(warm[:3],angle)
  # Native free-joint angular velocity/acceleration use body coordinates.
  assert np.array_equal(qvel[3:],source['source_native_qvel'][i,3:]) and np.array_equal(warm[3:],source['source_native_warmstart'][i,3:])
  assert np.array_equal(state[13:],source['source_state'][i,13:])
  for key in source:records[key].append(state if key=='query_state' else source[key][i])
  for key,value in [('query_native_qpos',qpos),('query_native_qvel',qvel),('query_native_warmstart',warm)]:records[key].append(value)
  row={k:anchor[k] for k in ['trace','id','source_family','start','index','reference_frame','split']};row.update(query=len(rows),anchor_query=i,yaw_degrees=angle,is_native_anchor=angle==0,use_native_coordinates=True,has_archived_next_label=angle==0);rows.append(row)
assert len(rows)==(1440 if expanded else 240)
f=a.output/'queries.npz';np.savez_compressed(f,**{k:np.stack(v) for k,v in records.items()})
plan=dict(contract='expanded_fit_native_world_yaw_symmetry_v1' if expanded else 'fit_native_world_yaw_symmetry_v1',queries=len(rows),angles_degrees=angles,rows=rows,sources=old['sources'],collection=old['collection'],manifest=old['manifest'],query_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source_query_plan_sha256=hashlib.sha256((a.source_queries/'query_plan.json').read_bytes()).hexdigest(),validation_arrays_loaded=False,optimizer_updates=0,training_approved=False,promotion_approved=False,scope='All predeclared fixed fit native anchors, global world rotations. Native translation qvel/qacc rotate; body angular/joint components unchanged. Original command references not rotated: terminations are not covariance tests. Source archived next is a label only atzero; other physical outputs must be measured.')
(a.output/'query_plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(dict(queries=len(rows),angles=angles,sha256=plan['query_sha256'])))
