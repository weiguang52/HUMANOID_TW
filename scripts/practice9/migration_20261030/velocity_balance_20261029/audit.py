from pathlib import Path
import json,yaml,sys,hashlib,torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
s=Path(__file__).parent; mode=sys.argv[1] if len(sys.argv)>1 else 'smoke'
old=Path('/root/gpufree-data/datasets/practice9/anchor_strength_20261028')
p=Path((s/f'actual_{mode}/final_checkpoint').read_text().strip()).parent
control_mode='smoke' if mode=='reload' else mode
q=Path((old/f'actual_{control_mode}/final_checkpoint').read_text().strip()).parent
read=lambda p:yaml.load(p.read_text(),Loader=yaml.BaseLoader)
x=read(p/'params/env.yaml');y=read(q/'params/env.yaml')
f=x['commands']['motion']['motion_file'];g=y['commands']['motion']['motion_file']
assert Path(f).read_bytes()==Path(g).read_bytes()
x['commands']['motion']['motion_file']=g
assert x['rewards']['motion_body_lin_vel']['weight']=='3.0' and y['rewards']['motion_body_lin_vel']['weight']=='2.0'
x['rewards']['motion_body_lin_vel']['weight']=y['rewards']['motion_body_lin_vel']['weight']
assert x==y
x=read(p/'params/agent.yaml');y=read(q/'params/agent.yaml')
assert x['algorithm']['lam']==y['algorithm']['lam']=='0.99'
x['algorithm']['lam']=y['algorithm']['lam']
x['run_name']=y['run_name']
assert x['experiment_name']=='/root/twrl_runs/velocity_balance_20261029/logs'
x['experiment_name']=y['experiment_name']
if mode=='reload':
 cp=Path((s/'actual_smoke/final_checkpoint').read_text().strip())
 assert x['load_run']==cp.parent.name and x['load_checkpoint']==cp.name and x['max_iterations']=='1'
 for k in ['load_run','load_checkpoint','max_iterations']: x[k]=y[k]
 assert (s/'actual_reload/initial_checkpoint').read_text()==(s/'actual_smoke/final_checkpoint').read_text()
assert x==y
for f in ['params/control_runtime.json','params/action_filter.json','params/contact_feedback.json']:
 assert read(p/f)==read(q/f),f
physics=read(p/'params/physics_slip.json');prior_physics=read(q/'params/physics_slip.json')
assert physics==prior_physics
x=json.loads((p/'policy_anchor.json').read_text());y=json.loads((q/'policy_anchor.json').read_text())
assert x['coefficient']==y['coefficient']==2.0
x['coefficient']=y['coefficient']
assert x==y
x=read(p/'params/temporal_anchor.json');y=read(q/'params/temporal_anchor.json')
assert x==y and x['target']=='student_mean'
w=json.loads((p/'physics_slip_completed.json').read_text())
assert w=={'completed_windows':{'smoke':48,'reload':24,'formal':12000}[mode],'unconsumed_samples':0}
a=torch.load(p/Path((s/f'actual_{mode}/final_checkpoint').read_text().strip()).name,map_location='cpu',weights_only=False)
assert a['iter']=={'smoke':27996,'reload':27996,'formal':28494}[mode]
for kind in ['actor_state_dict','critic_state_dict']:
 assert all(torch.isfinite(v).all() for v in a[kind].values())
assert a['actor_state_dict']['mlp.0.weight'].shape[1]==203
steps=sorted(set(float(v['step']) for v in a['optimizer_state_dict']['state'].values() if 'step' in v))
assert steps==[{'smoke':560040,'reload':560060,'formal':570000}[mode]],steps
e=EventAccumulator(str(p));e.Reload()
loss={k:[z.value for z in e.Scalars(k)] for k in e.Tags()['scalars'] if 'anchor' in k.lower() and 'loss' in k.lower()}
assert any('temporal' in k and any(v>0 for v in rows) for k,rows in loss.items())
out={'passed':True,'mode':mode,'run':str(p),'single_change':'Compared with scoped17/13: active17 PPO score/entropy; unchanged raw30 sampling and full30 KL/rewards/anchors/control','physics_windows':w,'Adam_steps':steps,'losses':loss,'quality_verified':False,'checkpoint_sha256':hashlib.sha256((p/Path((s/f'actual_{mode}/final_checkpoint').read_text().strip()).name).read_bytes()).hexdigest()}
import sys
sys.path.insert(0,'/root/gpufree-data/projects/HUMANOID_TW/scripts/rsl_rl')
from training_speed_blend import make_contract,audit_teacher_binding
from reference_speed_blend import candidate_weight,mix_actions
import numpy as np
torch.set_num_threads(1)
c=json.loads((p/'params/training_speed_blend.json').read_text())
assert c['version']==3 and c['candidate_action_scope']['scope']=='lower_body_waist'
scope=c['candidate_action_scope'];assert len(scope['candidate_joint_names'])==17 and len(scope['teacher_joint_names'])==13
assert c==make_contract(c['teacher_export'],json.loads((p/'policy_anchor.json').read_text())['teacher_checkpoint'],scope['action_joint_names'],scope['scope'],c['controlled_action_likelihood'])
assert audit_teacher_binding(c['teacher_export'],json.loads((p/'policy_anchor.json').read_text())['teacher_checkpoint'])['passed']
audit=json.loads((p/'training_speed_blend_completed.json').read_text())
assert audit['completed_steps']=={'smoke':48,'reload':24,'formal':12000}[mode]
assert audit['num_envs']==(1024 if mode=='formal' else 64)
assert audit['active_samples']+audit['zero_samples']==audit['num_envs']*audit['completed_steps']
assert audit['active_samples']>0 and audit['zero_samples']>0
teacher=torch.jit.load(c['teacher_export'],map_location='cpu').eval();z=np.load(p/'training_speed_blend_trace.npz')
assert len(z['raw_action'])==min(64,audit['completed_steps']) and z['teacher_observation'].shape[-1]==195
assert z['raw_action'].shape[-1]==30 and z['terminal'].shape==z['raw_action'].shape[:2]
assert all(np.isfinite(z[k]).all() for k in z.files)
errors={}
with torch.inference_mode():
 t=teacher(torch.from_numpy(z['teacher_observation'].copy()).flatten(0,1)).reshape(z['teacher_action'].shape)
 errors['teacher_action']=float((t-torch.from_numpy(z['teacher_action'])).abs().max())
 weight=candidate_weight(torch.from_numpy(z['reference_speed']),1.,.05,.1)
 errors['candidate_weight']=float((weight-torch.from_numpy(z['candidate_weight'])).abs().max())
 from candidate_action_scope import mix_scoped_actions
 mixed=mix_scoped_actions(torch.from_numpy(z['teacher_action']),torch.from_numpy(z['raw_action']),torch.from_numpy(z['candidate_weight']),scope['scope'],scope['action_joint_names'])
 errors['mixed_action']=float((mixed-torch.from_numpy(z['mixed_action'])).abs().max())
 assert errors['teacher_action']<1e-5 and errors['candidate_weight']<1e-6 and errors['mixed_action']==0,errors
 frozen=np.array([n in scope['teacher_joint_names'] for n in scope['action_joint_names']])
 assert np.array_equal(z['mixed_action'][...,frozen],z['teacher_action'][...,frozen])
 active=z['candidate_weight']>0
 assert active.any() and np.any(abs(z['raw_action'][active][:,frozen]-z['teacher_action'][active][:,frozen])>.001)
 out['teacher13_nonvacuous_active_raw_action_difference']=True
 out['teacher13_exact_all_captured_steps']=True
 out['candidate17_partition_verified']=True
 mask=z['candidate_weight']==0
 assert np.array_equal(z['mixed_action'][mask],z['teacher_action'][mask])
 if mode=='reload':
  previous=Path((s/'actual_smoke/final_checkpoint').read_text().strip()).parent
  assert c==json.loads((previous/'params/training_speed_blend.json').read_text())
from controlled_action_likelihood import make_contract as make_likelihood_contract, selected_indices
likelihood=json.loads((p/'params/controlled_action_likelihood.json').read_text())
assert likelihood==make_likelihood_contract(scope['action_joint_names'])==c['controlled_action_likelihood']
install=json.loads((p/'controlled_likelihood_install_audit.json').read_text())
assert install['enabled'] and install['contract']==likelihood and install['full30_KL_unchanged'] and install['raw30_sampling_unchanged']
a_activity=json.loads((p/'controlled_activity_completed.json').read_text())
assert a_activity['completed_steps']==audit['completed_steps'] and a_activity['captured_steps']==len(z['raw_action'])
activity=np.load(p/'controlled_activity_trace.npz')
assert activity['active'].shape==(*z['candidate_weight'].shape,1)
assert np.array_equal(activity['active'][...,0],z['candidate_weight']>0)
assert np.array_equal(activity['weight'],z['candidate_weight'])
assert activity['active'].any() and (~activity['active']).any()
if mode=='reload':
 previous=Path((s/'actual_smoke/final_checkpoint').read_text().strip()).parent
 assert likelihood==json.loads((previous/'params/controlled_action_likelihood.json').read_text())
for line in (s/f'actual_{mode}/code_sha256.txt').read_text().splitlines():
 expected, filename=line.split(maxsplit=1)
 assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
out['controlled_likelihood']={'contract':likelihood,'install_audit':install,'activity_audit':a_activity,'captured_current_gate_exact_mixture':True,'limits':'Actual simulator source-guarded install, all-step stale-gate checks and captured gate/mixture exact. Raw scores/gradients/storage pairing tested separately on CPU/CUDA; no full simulator score trace captured.'}
out['training_mixture_runtime']={**audit,'prefix_reconstruction_max_error':errors,'all_zero_weight_actions_exact_teacher':True,'scope':'Captured first64 steps only; does not claim full trajectory reconstruction or motion quality'}
out['single_change']='Only motion_body_lin_vel reward weight2 to3; anchor_strength parent otherwise equal'
out['minimum_normal_load']=5.
lookback=json.loads((p/'lookback_sampling_completed.json').read_text())
assert lookback['contract']['share']==0.5 and lookback['calls']>0
if mode=='formal': assert lookback['changed_calls']>0
out['lookback_runtime']=lookback
(s/f'{mode}_audit.json').write_text(json.dumps(out,indent=2))
if mode=='smoke':
 (s/'smoke_runtime_audit.json').write_text(json.dumps(out,indent=2))
 (s/'smoke_likelihood_audit.json').write_text(json.dumps(out,indent=2))
print(mode,'PASS',s.name,steps)
