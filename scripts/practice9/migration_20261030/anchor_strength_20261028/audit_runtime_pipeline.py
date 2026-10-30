from pathlib import Path
import sys,json,hashlib,numpy as np,torch
state=Path(sys.argv[1]);torch.set_num_threads(1)
partial='--partial' in sys.argv
if not partial:assert (state/'evaluation_status').read_text().strip()=='completed'
freeze=json.loads((state/'checkpoint_freeze.json').read_text());cp=Path(freeze['checkpoint'])
assert hashlib.sha256(cp.read_bytes()).hexdigest()==freeze['sha256']
student_path=cp.parent/'exported/policy.pt'
student=torch.jit.load(str(student_path),map_location='cpu').eval()
control=json.loads((cp.parent/'params/control_runtime.json').read_text())
contact=json.loads((cp.parent/'params/contact_feedback.json').read_text())
assert contact['mode']=='actual'
training_contract=json.loads((cp.parent/'params/training_speed_blend.json').read_text())
sys.path.insert(0,'/root/gpufree-data/projects/HUMANOID_TW/scripts/rsl_rl')
from controlled_action_likelihood import detach_training_contract,selected_indices
physical_contract,likelihood=detach_training_contract(training_contract)
assert likelihood==json.loads((cp.parent/'params/controlled_action_likelihood.json').read_text())
assert len(selected_indices(likelihood))==17
receipt=json.load(open(Path('/root/twrl_runs/load_aligned_20261027/recovery_20261028/equivalent_teacher_binding_audit.json')))
assert receipt['passed'] and receipt['original_export_bytes_recovered'] is False
immutable_teacher=receipt['new_immutable_teacher']
assert hashlib.sha256(Path(immutable_teacher).read_bytes()).hexdigest()==receipt['new_export_sha256']
from training_speed_blend import audit_teacher_binding
assert audit_teacher_binding(immutable_teacher,receipt['teacher_checkpoint'])['passed']
assert training_contract['teacher_export']==immutable_teacher and training_contract['teacher_export_sha256']==receipt['new_export_sha256']
rows=[];sources={};teacher_cache={}
files=[('retention',42,state/'retention/jobs.json')]+[(phase,seed,state/phase/'candidate'/('jobs'+str(seed)+'.json')) for phase in ['evaluation','independent'] for seed in [42,123,2026]]
for phase,seed,file in files:
 log_path=state/'retention/run.log' if phase=='retention' else state/phase/'candidate'/('seed'+str(seed)+'.log')
 log=log_path.read_text() if log_path.exists() else ''
 if not partial:assert Path(str(file)+'.done').read_text().strip()=='completed'
 for job in json.loads(file.read_text()):
  p=Path(job['output'])
  if partial and 'BATCH_COMPLETED '+str(p) not in log:continue
  assert 'BATCH_COMPLETED '+str(p) in log
  meta=json.loads(p.read_text());physical=np.load(p.with_suffix('.npz'));z=np.load(p.with_suffix('.policy_observations.npz'));blend=np.load(p.with_suffix('.blend.npz'))
  expected_contract=training_contract
  assert meta['training_speed_blend_contract']['contract']==expected_contract
  assert meta['checkpoint']==str(cp)
  names=physical['joint_names'].tolist();assert names==control['joint_names']
  assert set(blend.files)=={'speed_and_candidate_weight'}
  assert 'locomotion_damping_contract' not in meta and 'candidate_leg_filter' not in meta['policy_blend'] and 'candidate_hip_filter' not in meta['policy_blend']
  cfg=meta['action_filter'];assert cfg['alpha']==.5 and cfg['scope']=='ankles' and cfg['mode']=='absolute'
  selected=np.array(['ankle' in n for n in names]);assert selected.sum()==4 and cfg['selected_joints']==[n for n,s in zip(names,selected) if s]
  keys=z['observation_keys'].tolist();obs=z['observation_'+str(keys.index('policy'))][:,0].copy()
  assert obs.shape==(len(physical['terminal']),203) and int(z['base_policy_dimension'])==195 and np.isfinite(obs).all()
  base=z['observation_'+str(keys.index('contact_base_policy'))][:,0]
  assert np.array_equal(base,obs[:,:195])
  features=obs[:,195:].reshape(-1,2,4)
  assert np.isfinite(features).all() and np.any(features[:,:,0]>0)
  assert np.all((features[:,:,0]>=0)&(features[:,:,0]<=3))
  assert np.all(np.abs(features[:,:,1:3])<=5)
  assert np.all((features[:,:,3]>=0)&(features[:,:,3]<=5))
  unloaded=features[:,:,0]<=.05-1e-7
  assert np.all(features[:,:,1:][unloaded]==0)
  assert np.array_equal(z['action'][:,0],physical['action'])
  original_teacher_path=job['reference_speed_teacher_export']
  assert original_teacher_path==expected_contract['teacher_export']
  teacher_path=immutable_teacher
  if teacher_path not in teacher_cache:
   teacher_cache[teacher_path]=torch.jit.load(teacher_path,map_location='cpu').eval()
   sources[teacher_path]=hashlib.sha256(Path(teacher_path).read_bytes()).hexdigest()
  with torch.inference_mode():
   tensor=torch.from_numpy(obs);raw=student(tensor).numpy();teacher=teacher_cache[teacher_path](tensor[:,:195]).numpy()
  assert raw.shape==teacher.shape==(len(obs),30) and np.isfinite(raw).all() and np.isfinite(teacher).all()
  trace=blend['speed_and_candidate_weight'][:,0];speed,weight=trace[:,0],trace[:,1]
  assert np.isfinite(trace).all() and job['maximum_candidate_weight']==1. and job['blend_speed_low']==.05 and job['blend_speed_high']==.1
  u=np.clip((speed-.05)/.05,0,1);expected_weight=u*u*(3-2*u)
  assert np.allclose(weight,expected_weight,atol=2e-7,rtol=1e-6)
  mixed=teacher+(raw-teacher)*weight[:,None]
  mixed[weight==0]=teacher[weight==0];mixed[weight==1]=raw[weight==1]
  scope=training_contract.get('candidate_action_scope')
  if scope is not None:
   assert meta['policy_blend']['candidate_action_scope']==scope and scope['action_joint_names']==names
   frozen=np.array([n in scope['teacher_joint_names'] for n in names]);assert frozen.sum()==13 and len(scope['candidate_joint_names'])==17
   mixed[:,frozen]=teacher[:,frozen]
  output=[];previous=None
  for t,action in enumerate(mixed):
   final=action.copy()
   if previous is not None:final[selected]=.5*action[selected]+.5*previous[selected]
   output.append(final);previous=None if physical['terminal'][t] else final.copy()
  output=np.stack(output);recorded=physical['action'];delta=float(np.max(np.abs(output-recorded)))
  passed=bool(np.allclose(output,recorded,atol=1e-5,rtol=1e-5));assert passed,(str(p),delta)
  rows.append({'phase':phase,'eval_seed':seed,'path':str(p),'steps':len(obs),'all_pipeline_actions_match':passed,'cpu_gpu_max_action_delta':delta,'resets':int(physical['terminal'].sum()),'candidate_active_steps':int(np.count_nonzero(weight>0))})
if not partial:assert len(rows)==65
out={'complete':len(rows)==65,'expected':65,'observed':len(rows),'all_observed_pass':bool(rows) and all(v['all_pipeline_actions_match'] for v in rows),'rows':rows,'checkpoint':str(cp),'checkpoint_sha256':freeze['sha256'],'student_jit_sha256':hashlib.sha256(student_path.read_bytes()).hexdigest(),'teacher_jit_sha256':sources,'teacher_reexport_receipt':str(Path('/root/twrl_runs/load_aligned_20261027/recovery_20261028/equivalent_teacher_binding_audit.json')),'original_teacher_export_bytes_restored':False,'teacher_reconstruction_uses_verified_equivalent_export':True,'scope':'Full recorded observations -> student/teacher NN -> reference-speed gate -> ankle IIR -> delivered action, including resets. Does not establish walking quality or visual acceptance.'}
(state/('runtime_pipeline_partial_audit.json' if partial else 'runtime_pipeline_audit.json')).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
