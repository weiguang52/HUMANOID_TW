"""Independent native quarter-turn archive checks; retain every precision failure."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from supermimic_yaw_transform import rotate_world_state,state_metrics
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','collection','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
source_rows=json.loads((a.collection/'results.json').read_text())['rows'];assert len(source_rows)==32
pools={};reports=[];evidence=[]
for turns in [0,1,2]:
 root=a.root/('native_'+str(turns)+('_complete' if turns==1 else ''));plan=json.loads((root/'plan.json').read_text());results=json.loads((root/'results.json').read_text())
 assert plan['yaw_degrees']==90*turns and plan['refresh_command_cache'] and not plan['smoke'] and plan['phase']=='cold'
 assert plan['native_replay_atol']==1e-5 and plan['state_write_atol']==2e-6 and plan['context_atol']==1e-7
 assert [x['file'] for x in results['rows']]==[x['trace'] for x in source_rows]
 pieces={k:[] for k in ['state','actual','target','context']};per_trace=[]
 for row,replayed in zip(source_rows,results['rows']):
  src=a.collection/row['trace'];assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
  with np.load(src,allow_pickle=False) as z:original={k:z[k].copy() for k in ['state','context','next_state']}
  path=root/replayed['trace'];assert path.resolve().is_relative_to(root.resolve())
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert all(np.isfinite(x).all() for x in d.values())
  idx=np.asarray(sorted({0,len(original['state'])//2,len(original['state'])-1}));np.testing.assert_array_equal(d['index'],idx)
  query=rotate_world_state(torch.tensor(original['state'][idx]),turns).numpy();target=rotate_world_state(torch.tensor(original['next_state'][idx]),turns).numpy()
  write_error=np.abs(d['restored_state']-query).max(axis=1);write_bad=idx[write_error>2e-6]
  np.testing.assert_allclose(d['context'],original['context'][idx],atol=1e-7,rtol=0)
  np.testing.assert_array_equal(d['recorded_next_state'][:,:3],target[:,:3]);np.testing.assert_array_equal(d['recorded_next_state'][:,7:],target[:,7:])
  transform_error=float(np.max(np.abs(d['recorded_next_state'][:,3:7]-target[:,3:7])))
  assert transform_error<=2e-7  # Same float32 quaternion oracle bound as model transform audit, separate from native tolerances.
  error=np.abs(d['actual_next_state']-target).max(axis=1);bad=idx[error>1e-5]
  per_trace.append(dict(recorded_quaternion_transform_max_error=transform_error,state_write_failure_indices=write_bad.tolist(),max_state_write_error=float(write_error.max()),trace=row['trace'],split=row['split'],indices=idx.tolist(),failed_indices=bad.tolist(),max_state_error=float(error.max()),terminated_indices=idx[d['terminated'].astype(bool)].tolist()))
  evidence.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  for k,v in [('state',d['restored_state']),('actual',d['actual_next_state']),('target',target),('context',d['context'])]:pieces[k].append(v)
 pools[turns]={k:np.concatenate(v) for k,v in pieces.items()};assert len(pools[turns]['state'])==96
 actual=torch.tensor(pools[turns]['actual']);target=torch.tensor(pools[turns]['target'])
 equiv=rotate_world_state(torch.tensor(pools[0]['actual']),turns)
 reports.append(dict(state_write_failed_states=sum(len(x['state_write_failure_indices']) for x in per_trace),max_state_write_error=max(x['max_state_write_error'] for x in per_trace),yaw_degrees=90*turns,states=96,failed_states=sum(len(x['failed_indices']) for x in per_trace),all_within_original_native_atol=all(not x['failed_indices'] for x in per_trace),native_vs_transformed_archive=state_metrics(actual,target),native_vs_rotated_zero_replay=state_metrics(actual,equiv),rows=per_trace))
np.savez_compressed(a.output/'native_queries.npz',**{str(turn)+'_'+k:v for turn,pool in pools.items() for k,v in pool.items()})
out=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),reports=reports,evidence=evidence,archive_invariants_passed=True,scope='All288 fixed native queries; Every source/current input error is reported at unchanged2e-6 tolerance, including failures. Physical quarter-turn consistency only at tested states, no universal symmetry/Markov/control proof; original terminal flags may change because global reference goals are not rotated, with autoreset disabled.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in reports]),flush=True)
