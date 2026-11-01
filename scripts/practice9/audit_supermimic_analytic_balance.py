"""Audit declared real-loop targets, provenance and past-only contact timing."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np
from supermimic_contact_balance import ContactBalance
from supermimic_residual_contact_balance import ResidualContactBalance
from supermimic.config.tw.velocity_limits import LIMITS,PHYSICS_CONTRACT
p=argparse.ArgumentParser(description=__doc__)
for k in ['plan','run','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();plan=json.loads(a.plan.read_text());info=json.loads((a.run/'input.json').read_text());assert info['plan_sha256']==hashlib.sha256(a.plan.read_bytes()).hexdigest();assert info['asset_sha256']==hashlib.sha256(a.asset.read_bytes()).hexdigest();assert info['physics']==PHYSICS_CONTRACT and info['seed']==42
assert info.get('feedback_contract')==plan.get('feedback_contract')
root=Path('/root/gpufree-data/datasets/practice9');train=root/'training_speed_blend_20261026/train.json';held=root/'independent_holdout_20261025/test.json'
for f,key in [(train,'train_manifest_sha256'),(held,'fixed7_manifest_sha256')]:assert hashlib.sha256(f.read_bytes()).hexdigest()==plan[key]
original={r['id']:r for r in json.loads(train.read_text())['motions']};test=json.loads(held.read_text())['motions'];assert not ({r['source_family'] for r in plan['motions']} & {r['source_family'] for r in test});assert not ({r['id'] for r in plan['motions']} & {r['id'] for r in test})
for r in plan['motions']:assert all(r[k]==original[r['id']][k] for k in ['id','file','frames','source_family'])
assert not ({r['source_family'] for r in plan['motions'] if r['split']=='fit'} & {r['source_family'] for r in plan['motions'] if r['split']=='validation'})
names=info['state_joint_names'];order=[names.index(n) for n in info['action_joint_names']];scale=np.array(info['scale']);offset=np.array(info['offset']);caps=np.array([LIMITS[n] for n in names]);model=mujoco.MjModel.from_xml_path(str(a.asset));controller=(ResidualContactBalance if plan.get('feedback_contract')=='residual_about_bounded_nominal_v2' else ContactBalance)(model,names);refs={r['id']:r for r in plan['motions']};rows=json.loads((a.run/'results.json').read_text())['rows'];expected={(r['id'],int((r['frames']-1)*f)) for r in (plan['motions'][:1] if info['smoke'] else plan['motions']) for f in ([0.] if info['smoke'] else plan['phase_fractions'])};assert len(rows)==len(expected) and {(r['id'],r['frame']) for r in rows}==expected;reports=[]
for row in rows:
 r=refs[row['id']];assert row['split']==r['split'] and row['source_family']==r['source_family'];refpath=Path(r['file']);assert hashlib.sha256(refpath.read_bytes()).hexdigest()==r['sha256'];f=a.run/row['trace'];logpath=a.run/row['control_file'];assert f.resolve().is_relative_to(a.run.resolve()) and logpath.resolve().is_relative_to(a.run.resolve());assert hashlib.sha256(f.read_bytes()).hexdigest()==row['trace_sha256'];assert hashlib.sha256(logpath.read_bytes()).hexdigest()==row['control_sha256'];logs=json.loads(logpath.read_text());n=row['steps'];assert 0<n<=row['planned_steps']==min(3 if info['smoke'] else plan['max_steps'],r['frames']-row['frame']-1);assert len(logs)==n
 with np.load(f) as z,np.load(refpath) as ref:
  assert all(len(z[k])==n and np.isfinite(z[k]).all() for k in z.files);assert np.array_equal(z['next_state'][:-1],z['state'][1:]);done=(z['terminated']|z['truncated']).reshape(-1);assert not done[:-1].any() and (n==row['planned_steps'] or done[-1]);assert bool(done[-1])==bool(row['termination_reasons']);assert np.all(abs(z['next_state'][:,43:73])<=caps+1e-5);inverse=float(abs(z['action']*scale+offset-z['target_joint_pos'][:,order]).max());assert inverse<=1e-6;maximum=0.;idx=[ref['joint_names'].tolist().index(n) for n in names];base=ref['body_names'].tolist().index('base_link')
  for i,log in enumerate(logs):
   assert log['action_step']==i+1 and log['contact_source_step']==i
   if i:assert log['prior_feet']==logs[i-1]['post_feet']
   else:assert all(not f['contacts'] for f in log['prior_feet'].values())
   frame=int(z['reference_frame'][i,0]);assert frame==row['frame']+i;nominal=ref['joint_pos'][frame,idx];assert abs(z['nominal_joint_pos'][i]-nominal).max()<1e-6
   target,detail=controller.target(z['state'][i],nominal,ref['root_quat_wxyz'][frame],ref['body_ang_vel_w'][frame,base],log['prior_feet'],info['arm']=='contact_balance');error=float(abs(target-z['target_joint_pos'][i]).max());maximum=max(maximum,error);assert error<=1e-6
   inc=z['target_joint_pos'][i,controller.lower]-z['state'][i,13:43][controller.lower];assert abs(inc).max()<=.060001;assert abs(z['target_joint_pos'][i,14:]-nominal[14:]).max()<=1e-6
 reports.append(dict(id=row['id'],frame=row['frame'],steps=n,target_recompute_max_error=maximum,inverse_target_max_error=inverse,past_only_contacts=True))
a.output.write_text(json.dumps(dict(passed=True,arm=info['arm'],smoke=info['smoke'],scope='Declared target reconstruction, timing/provenance and trajectory bounds only; no physical stability or policy acceptance.',rows=reports),indent=2)+'\n');print('ANALYTIC_BALANCE_AUDIT_PASSED',len(rows))
