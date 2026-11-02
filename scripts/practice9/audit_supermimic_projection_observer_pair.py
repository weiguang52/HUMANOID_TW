"""Audit complete native substep observations and localize repeat differences."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
for key in ['collection','first','second','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
source=json.loads((a.collection/'results.json').read_text())['rows'];assert len(source)==32
plans=[json.loads((d/'plan.json').read_text()) for d in [a.first,a.second]]
assert plans[0]==plans[1] and plans[0]['cold_all_frames'] and not plans[0]['smoke']
assert len(plans[0]['source_files'])==32
assert plans[0]['restore_native_coordinates'] and plans[0]['restore_native_warmstart']
points=sum(r['steps'] for r in source);arrays=[];metadata=[]
for folder in [a.first,a.second]:
 rows=json.loads((folder/'results.json').read_text())['rows'];assert len(rows)==32
 assert [x['file'] for x in rows]==[x['trace'] for x in source]
 for row,replay,bound in zip(source,rows,plans[0]['source_files']):
  path=a.collection/row['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']==bound['sha256']
  assert replay['replayed_frames']==row['steps']
  with np.load(folder/replay['trace'],allow_pickle=False) as z:assert np.array_equal(z['index'],np.arange(row['steps']))
 meta=json.loads((folder/'projection_observer.json').read_text());metadata.append(meta)
 assert meta['input_mutation_checks_passed'] and not meta['physics_changed']
 with np.load(folder/'projection_substeps.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 assert data['pre_velocity'].shape==data['post_velocity'].shape==(points*4,36)
 assert data['qpos'].shape==(points*4,37) and data['dense_mass'].shape==(points*4,36,36)
 assert meta['substeps']==points*4 and meta['active_substeps']==int(data['active'].sum())
 assert all(len(x)==points*4 and np.isfinite(x).all() for x in data.values())
 assert np.array_equal(data['pre_velocity'][~data['active']],data['post_velocity'][~data['active']])
 assert set(data['iterations'].tolist())<={0,64,128,192,256}
 assert np.all(data['iterations'][~data['active']]==0)
 arrays.append(data)
assert metadata[0]['observer_sha256']==metadata[1]['observer_sha256']
assert metadata[0]['physics_source_sha256']==metadata[1]['physics_source_sha256']
x,y=arrays
errors={k:np.abs(x[k]-y[k]) for k in ['pre_velocity','post_velocity','qpos','sparse_mass','dense_mass']}
pre=errors['pre_velocity'].max(axis=1);post=errors['post_velocity'].max(axis=1)
rows=[];offset=0
for row in source:
 for index in range(row['steps']):
  start=offset+index*4;end=start+4
  if max(pre[start:end].max(),post[start:end].max())>1e-5:
   indices=np.flatnonzero((pre[start:end]>1e-5)|(post[start:end]>1e-5));j=start+int(indices[0])
   rows.append(dict(trace=row['trace'],index=index,first_over_threshold_substep=int(indices[0]),pre_error=float(pre[j]),post_error=float(post[j]),pre_already_over_threshold=bool(pre[j]>1e-5),active_first=bool(x['active'][j]),active_second=bool(y['active'][j]),iterations_first=int(x['iterations'][j]),iterations_second=int(y['iterations'][j]),qpos_error=float(errors['qpos'][j].max()),mass_error=float(errors['sparse_mass'][j].max())))
 offset+=row['steps']*4
assert offset==points*4
report=dict(traces=32,points=points,substeps=points*4,coverage_verified=True,active_substeps=[int(z['active'].sum()) for z in arrays],iteration_histograms=[{str(k):int(v) for k,v in zip(*np.unique(z['iterations'],return_counts=True))} for z in arrays],active_flag_differences=int(np.count_nonzero(x['active']!=y['active'])),iteration_differences=int(np.count_nonzero(x['iterations']!=y['iterations'])),max_differences={k:float(e.max()) for k,e in errors.items()},first_substep_max_differences={k:float(e[::4].max()) for k,e in errors.items()},transitions_over_1e_minus5=len(rows),transitions=rows,scope='Read-only substep localization; pre_velocity is after native acceleration integration, before velocity projection. No causal proof about other solver kernels; no policy acceptance.')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='transitions'}))
