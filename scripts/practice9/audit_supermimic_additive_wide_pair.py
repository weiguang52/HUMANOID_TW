"""Verify matched complete-objective dynamic action-response calibration."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--updates',type=int,choices=[4,1000],required=True);p.add_argument('--reference-control',type=Path,required=True);a=p.parse_args()
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
paths={arm:a.root/arm/'fit' for arm in ('control','candidate')};models={k:torch.load(p/'world_model_calibrated.pt',map_location='cpu',weights_only=False) for k,p in paths.items()};ref=torch.load(a.reference_control,map_location='cpu',weights_only=False);c=models['control'];d=models['candidate'];assert a.updates!=4 or (equal(c['world_model'],ref['world_model']) and equal(c['optimizer'],ref['optimizer']))
if a.updates==1000:assert json.loads((a.root/'smoke4/fit_integrity.json').read_text())['passed']
frozen=[k for k in c['world_model'] if not k.startswith('net.')];assert len(frozen)==22 and all(torch.equal(c['world_model'][k],d['world_model'][k]) for k in frozen)
plans=[models[k]['calibration_plan'] for k in ('control','candidate')];assert {k for k in set(plans[0])|set(plans[1]) if plans[0].get(k)!=plans[1].get(k)}=={'additional_wide_root_enabled'}
for arm,m in models.items():
 plan=m['calibration_plan'];assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==2176 and plan['actual_prediction_points_total']==2176*a.updates
 assert len(m['optimizer']['state'])==8 and all(float(s['step'])==2000+a.updates for s in m['optimizer']['state'].values())
 assert plan['static_step_guard'] is False and len(plan['motion_action_response_input']['fit_files'])==192
 center_h=json.loads((paths[arm]/'motion_action_center_history.json').read_text());assert len(center_h)==a.updates and all(len(x['indices'])==16 and all(0<=i<895 for i in x['indices']) for x in center_h)
 assert plan['motion_action_response_enabled'] is True and len(plan['motion_action_center_input']['fit_files'])==48
 h=json.loads((paths[arm]/'motion_action_history.json').read_text());assert len(h)==a.updates and all(len(x['indices'])==16 and all(0<=i<2880 for i in x['indices']) for x in h)
for file,key in [('motion_action_center_history.json','indices'),('motion_action_history.json','indices'),('static_step_guard_history.json','static_indices'),('native_replay_history.json','indices'),('native_h32_history.json','indices')]:
 histories=[json.loads((paths[k]/file).read_text()) for k in ('control','candidate')];assert [[x[key] for x in h] for h in histories][0]==[[x[key] for x in h] for h in histories][1]
assert plans[0]['additional_wide_root_enabled'] is False and plans[1]['additional_wide_root_enabled'] is True
assert all(x['motion_center_dense_enabled'] is True and x['motion_center_linear_enabled'] is True and x['resume_optimizer_steps']==2000 and x['resume_checkpoint_sha256']=='09a7410bdce35423b598f93a90d9fee8a24f2edeba62793ed06e177b6a270176' for x in plans)
assert all(x['motion_action_center_enabled'] is True and len(x['motion_dense_center_input']['fit_files'])==16 for x in plans)
for arm,bound in [('control',895),('candidate',895)]:
 generator=torch.Generator(device='cuda:0').manual_seed(61)
 history=json.loads((paths[arm]/'motion_action_center_history.json').read_text())
 for row in history:assert row['indices']==torch.randint(bound,(16,),device='cuda:0',generator=generator).cpu().tolist()
histories={arm:json.loads((paths[arm]/'wide_root_sample_history.json').read_text()) for arm in ['control','candidate']}
for field in ['original_indices','wide_level_indices','anchor_indices','axis_indices']:assert [r[field] for r in histories['control']]==[r[field] for r in histories['candidate']]
for arm in ['control','candidate']:
 g=torch.Generator(device='cuda:0').manual_seed(52);lg=torch.Generator(device='cuda:0').manual_seed(62);assert len(histories[arm])==a.updates
 for u,row in enumerate(histories[arm],1):
  old=torch.randint(576,(8,),device='cuda:0',generator=g);level=torch.randint(5,(8,),device='cuda:0',generator=lg);used=old
  assert row=={'update':u,'original_indices':old.cpu().tolist(),'wide_level_indices':level.cpu().tolist(),'used_indices':used.cpu().tolist(),'anchor_indices':(old//12).cpu().tolist(),'axis_indices':(old%6).cpu().tolist()}
 assert plans[0]['wide_root_velocity_input']['validation_arrays_loaded'] is False and len(plans[0]['wide_root_velocity_input']['files'])==480
for arm in ['control','candidate']:
 h=json.loads((paths[arm]/'additional_wide_history.json').read_text());g=torch.Generator(device='cuda:0').manual_seed(63);assert len(h)==a.updates
 for u,row in enumerate(h,1):
  old=torch.tensor(histories[arm][u-1]['original_indices'],device='cuda:0');level=torch.randint(3,(8,),device='cuda:0',generator=g)+2;idx=(old//12)*30+level*6+old%6
  assert row['update']==u and row['original_indices']==old.cpu().tolist() and row['wide_level_indices']==level.cpu().tolist() and row['used_indices']==idx.cpu().tolist() and row['half_weighted_loss']>=0
extra=[json.loads((paths[arm]/'additional_wide_history.json').read_text()) for arm in ['control','candidate']]
for k in ['original_indices','wide_level_indices','used_indices']:assert [x[k] for x in extra[0]]==[x[k] for x in extra[1]]
assert all(x['wide_root_velocity_enabled'] is False for x in plans)
resume=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/shared_hidden_moment_pair_v1/control/fit/world_model_calibrated.pt');assert hashlib.sha256(resume.read_bytes()).hexdigest()==plans[0]['resume_checkpoint_sha256']
parent=torch.load(resume,map_location='cpu',weights_only=False)
assert all(torch.equal(c['world_model'][k],parent['world_model'][k]) for k in frozen)
out={'passed':True,'updates':a.updates,'control_weights_optimizer_bitexact':True if a.updates==4 else None,'control_repeat_scope':'Repeated4 updates only; formal arms verify resume hash, exact paired amplitudes/anchors/axes, paired samplers and budgets','frozen_tensors':22,'adam_states_verified':8,'original_paired_indices_equal':True,'center_indices_replay_verified':True,'single_variable_verified':True,'points_per_update':2176,'policy_acceptance':False,'checkpoint_sha256':{k:hashlib.sha256((p/'world_model_calibrated.pt').read_bytes()).hexdigest() for k,p in paths.items()}};(a.root/'fit_integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
