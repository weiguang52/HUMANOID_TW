"""Independent structural/repeat audit for native rollout-neighborhood labels."""
import argparse,hashlib,json,ast
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--queries',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.queries/'query_plan.json').read_text());meta=json.loads((a.collection/'plan.json').read_text());out=a.collection/'independent_audit.json';assert not out.exists()
assert meta['query_plan_sha256']==hashlib.sha256((a.queries/'query_plan.json').read_bytes()).hexdigest()
assert meta['queries_sha256']==plan['query_sha256']==hashlib.sha256((a.queries/'queries.npz').read_bytes()).hexdigest()
fit_reference_initialization_verified=False
if 'fit_only_reference_ids' in meta:
 manifest=json.loads(Path(plan['manifest']).read_text());fit=[x for x in manifest['motions'] if x['split']=='fit'];assert len(fit)==4
 assert meta['manifest_sha256']==hashlib.sha256(Path(plan['manifest']).read_bytes()).hexdigest()
 assert meta['fit_only_reference_ids']==[x['id'] for x in fit]
 assert meta['fit_reference_files']==[dict(file=x['file'],sha256=x['sha256']) for x in fit]
 for x in fit:assert hashlib.sha256(Path(x['file']).read_bytes()).hexdigest()==x['sha256']
 fit_reference_initialization_verified=True
with np.load(a.queries/'queries.npz',allow_pickle=False) as z:q={k:z[k].copy() for k in z.files}
with np.load(a.collection/'transitions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
contract=plan.get('contract','rollout_neighborhood_v1');assert contract in ['rollout_neighborhood_v1','fit_dynamic_root_pose_probe_v1','fit_dynamic_root_pose_full_v1','fit_native_world_yaw_symmetry_v1']
assert meta.get('query_contract','rollout_neighborhood_v1')==contract
selected=plan['rows'][:{'rollout_neighborhood_v1':4,'fit_dynamic_root_pose_probe_v1':7,'fit_dynamic_root_pose_full_v1':25,'fit_native_world_yaw_symmetry_v1':5}[contract]] if meta['smoke'] else plan['rows'];indices=[x['query'] for x in selected]
def native_anchor(row):return bool(row['is_native_anchor']) if 'is_native_anchor' in row else row['depth']==0
expected=[(rep,warm,i) for rep in range(2) for warm in [True,False] for i in indices]
actual=list(zip(d['repeat'].tolist(),d['warm_source'].tolist(),d['query'].tolist()));assert actual==expected and len(actual)==meta['planned_records']
assert all(len(v)==len(actual) for v in d.values());assert all(np.isfinite(v).all() for v in d.values())
for key in ['state','next_state']:assert d[key].shape==(len(actual),73)
qi=d['query'];assert np.array_equal(d['action'],q['action'][qi])
if 'query_native_qpos' in q:
 assert np.array_equal(d['native_qpos'],q['query_native_qpos'][qi]) and np.array_equal(d['native_qvel'],q['query_native_qvel'][qi])
contract=Path(__file__).resolve().parents[2]/'third_party/supermimic_mjlab/src/supermimic/config/tw/contract.py'
groups={}
for node in ast.parse(contract.read_text()).body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['LEG_MAJOR_JOINTS','FOOT_JOINTS','WAIST_JOINTS','ARM_JOINTS','NECK_JOINTS']:groups[node.targets[0].id]=ast.literal_eval(node.value)
limit_source=contract.with_name('velocity_limits.py')
limit_node=next(node.value for node in ast.parse(limit_source.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LIMITS' for t in node.targets))
limits={name:ast.literal_eval(pair.elts[1]) for pair in limit_node.generators[0].iter.elts for name in groups[pair.elts[0].id]}
mapping=json.loads((Path(plan['collection'])/'control_mapping.json').read_text());caps=np.array([limits[n] for n in mapping['state_joint_names']],dtype=np.float32)
input_excess=np.maximum(np.abs(q['query_state'][qi,43:73])-caps,0).max(1)
np.testing.assert_allclose(input_excess,d['input_speed_excess'],atol=1e-7,rtol=0)
assert (np.abs(d['next_state'][:,43:73])<=caps+1e-5).all()

write=np.max(np.abs(d['state']-q['query_state'][qi]),axis=1);context=np.max(np.abs(d['context']-q['context'][qi]),axis=1)
np.testing.assert_allclose(write,d['state_write_error'],rtol=0,atol=0);np.testing.assert_allclose(context,d['context_error'],rtol=0,atol=0)
lookup={key:i for i,key in enumerate(actual)};repeat_errors=[];warm_errors=[];anchor_errors=[]
for i in indices:
 for warm in [True,False]:
  x,y=[lookup[(rep,warm,i)] for rep in [0,1]]
  assert np.array_equal(d['state'][x],d['state'][y]) and np.array_equal(d['context'][x],d['context'][y])
  assert np.array_equal(d['native_qpos'][x],d['native_qpos'][y]) and np.array_equal(d['native_qvel'][x],d['native_qvel'][y])
  assert np.array_equal(d['native_warmstart'][x],d['native_warmstart'][y])
  err=float(np.max(np.abs(d['next_state'][x]-d['next_state'][y])));repeat_errors.append(err)
  assert d['terminated'][x]==d['terminated'][y] and d['truncated'][x]==d['truncated'][y]
 for rep in [0,1]:
  x,y=[lookup[(rep,warm,i)] for warm in [True,False]]
  warm_errors.append(float(np.max(np.abs(d['next_state'][x]-d['next_state'][y]))))
  if native_anchor(plan['rows'][i]):
   assert np.array_equal(d['native_qpos'][x],q['source_native_qpos'][i]) and np.array_equal(d['native_qvel'][x],q['source_native_qvel'][i])
   anchor_errors.append(float(np.max(np.abs(d['next_state'][x]-q['source_next_state'][i]))))
 for warm in [True,False]:
  x=lookup[(0,warm,i)];warm_key='query_native_warmstart' if 'query_native_warmstart' in q else 'source_native_warmstart';target=q[warm_key][i] if warm else np.zeros(36,dtype=q[warm_key].dtype)
  assert np.array_equal(d['native_warmstart'][x],target)
cone=meta.get('diagnostic_cone','pyramidal');assert cone in ['pyramidal','elliptic']
assert meta['mujoco_after']['cone']==cone
archived_same_physics=cone=='pyramidal' and meta['mujoco_after']['iterations']==meta['mujoco_before']['iterations']
checks=dict(state_write=bool((write<=2e-6).all()),context_write=bool((context<=1e-7).all()),repeat_native=bool(max(repeat_errors)<=1e-5),archived_native_anchors=bool(max(anchor_errors)<=1e-5) if archived_same_physics else None,input_speed_caps=bool((d['input_speed_excess']<=1e-5).all()))
result=dict(archived_anchor_gate_applicable=archived_same_physics,diagnostic_cone=cone,fit_reference_initialization_verified=fit_reference_initialization_verified,passed=all(v for v in checks.values() if v is not None),checks=checks,smoke=meta['smoke'],queries=len(indices),records=len(actual),max_state_write_error=float(write.max()),max_context_error=float(context.max()),max_repeat_error=max(repeat_errors),max_archived_anchor_error=max(anchor_errors),max_warmstart_sensitivity=max(warm_errors),max_input_joint_range_excess=float(d['input_joint_range_excess'].max()),terminated_records=int(d['terminated'].sum()),transitions_sha256=hashlib.sha256((a.collection/'transitions.npz').read_bytes()).hexdigest(),training_approved=False,promotion_approved=False,scope='Structural/write/identical augmented-state repeat gates and archived anchors only. Warmstart sensitivity separately reported, not proof73-state Markov. Joint position range excess retained as advisory; all terminations kept. No physical-control acceptance.')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
