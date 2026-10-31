"""Independently verify reference-PD collection structure; no policy acceptance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from supermimic.config.tw.velocity_limits import LIMITS
p=argparse.ArgumentParser(description=__doc__)
for key in ['plan','collection','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
plan=json.loads(a.plan.read_text());root=a.collection
info=json.loads((root/'input.json').read_text());assert info['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest()
mapping=json.loads((root/'control_mapping.json').read_text());assert mapping['clip']=='None' and mapping['step_dt']==.02
names=mapping['state_joint_names'];assert len(names)==len(set(names))==30
order=[names.index(x) for x in mapping['action_joint_names']];assert sorted(order)==list(range(30))
offset=np.asarray(mapping['offset']);scale=np.asarray(mapping['scale']);caps=np.asarray([LIMITS[x] for x in names])
rows=json.loads((root/'results.json').read_text())['rows'];motions=plan['motions'][:2] if info['smoke'] else plan['motions']
expected={(m['id'],int((m['frames']-1)*f)) for m in motions for f in ([0.] if info['smoke'] else plan['phase_fractions'])}
assert len(rows)==len(expected) and {(x['id'],x['frame']) for x in rows}==expected
base=Path('/root/gpufree-data/datasets/practice9')
train=base/'training_speed_blend_20261026/train.json';held=base/'independent_holdout_20261025/test.json'
assert hashlib.sha256(train.read_bytes()).hexdigest()==plan['train_manifest_sha256']
assert hashlib.sha256(held.read_bytes()).hexdigest()==plan['fixed7_manifest_sha256']
held_rows=json.loads(held.read_text())['motions'];held_ids={x['id'] for x in held_rows};held_families={x['source_family'] for x in held_rows}
seen=set();selected=[]
for x in sorted(json.loads(train.read_text())['motions'],key=lambda x:x['id']):
 if x.get('category')!='walking' or x['source_family'] in seen:continue
 seen.add(x['source_family']);selected.append(x)
 if len(selected)==8:break
assert [x['id'] for x in selected]==[x['id'] for x in plan['motions']]
assert all(x['id'] not in held_ids and x['source_family'] not in held_families for x in plan['motions'])
by_id={m['id']:m for m in motions};reports=[]
for row in rows:
 m=by_id[row['id']];assert row['split']==m['split'] and row['source_family']==m['source_family'] and not row['learned_policy']
 refpath=Path(m['file']);assert hashlib.sha256(refpath.read_bytes()).hexdigest()==m['sha256']==row['reference_sha256']
 path=root/row['trace'];assert path.resolve().is_relative_to(root.resolve());assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 n=row['steps'];assert 0<n<=row['planned_steps'];assert all(len(v)==n and np.isfinite(v).all() for v in d.values())
 for k,w in [('state',73),('next_state',73),('action',30),('context',34),('desired_joint_pos',30),('terminated',1),('truncated',1)]:assert d[k].shape==(n,w)
 assert np.array_equal(d['next_state'][:-1],d['state'][1:])
 done=(d['terminated']|d['truncated']).reshape(-1);assert not done[:-1].any()
 assert n==row['planned_steps'] or done[-1]
 assert bool(done[-1])==bool(row['termination_reasons'])
 assert np.all(np.abs(d['next_state'][:,43:73])<=caps+1e-5)
 inverse=float(np.max(np.abs(d['action']*scale+offset-d['desired_joint_pos'][:,order])));assert inverse<=1e-6
 with np.load(refpath,allow_pickle=False) as ref:
  refnames=ref['joint_names'].tolist();idx=[refnames.index(x) for x in names]
  expected_target=ref['joint_pos'][row['frame']:row['frame']+n][:,idx]
  timing=float(np.max(np.abs(d['desired_joint_pos']-expected_target)));assert timing<=1e-6,timing
  initial=float(np.max(np.abs(d['state'][0,13:43]-expected_target[0])));assert initial<=1e-6
 reports.append(dict(trace=row['trace'],steps=n,terminated=bool(done[-1]),target_inverse_max_error=inverse,reference_target_max_error=timing,initial_joint_max_error=initial))
assert not ({m['source_family'] for m in motions if m['split']=='fit'} & {m['source_family'] for m in motions if m['split']=='validation'})
a.output.write_text(json.dumps(dict(passed=True,smoke=info['smoke'],rows=reports,scope='Reference timing/mapping, finite native traces, caps and original terminal records only; no learned walking, physical feasibility or policy acceptance'),indent=2))
print('MOTION_PD_AUDIT_PASSED',len(reports))
