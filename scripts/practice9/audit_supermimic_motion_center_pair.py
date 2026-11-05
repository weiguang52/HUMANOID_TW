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
paths={arm:a.root/arm/'fit' for arm in ('control','candidate')};models={k:torch.load(p/'world_model_calibrated.pt',map_location='cpu',weights_only=False) for k,p in paths.items()};ref=torch.load(a.reference_control,map_location='cpu',weights_only=False);c=models['control'];d=models['candidate'];assert equal(c['world_model'],ref['world_model']) and equal(c['optimizer'],ref['optimizer'])
frozen=[k for k in c['world_model'] if not k.startswith('net.')];assert len(frozen)==22 and all(torch.equal(c['world_model'][k],d['world_model'][k]) for k in frozen)
plans=[models[k]['calibration_plan'] for k in ('control','candidate')];assert {k for k in set(plans[0])|set(plans[1]) if plans[0].get(k)!=plans[1].get(k)}=={'motion_action_center_enabled'}
for arm,m in models.items():
 plan=m['calibration_plan'];assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==2144 and plan['actual_prediction_points_total']==2144*a.updates
 assert len(m['optimizer']['state'])==8 and all(float(s['step'])==a.updates for s in m['optimizer']['state'].values())
 assert plan['static_step_guard'] is False and len(plan['motion_action_response_input']['fit_files'])==192
 center_h=json.loads((paths[arm]/'motion_action_center_history.json').read_text());assert len(center_h)==a.updates and all(len(x['indices'])==16 and all(0<=i<48 for i in x['indices']) for x in center_h)
 assert plan['motion_action_response_enabled'] is True and len(plan['motion_action_center_input']['fit_files'])==48
 h=json.loads((paths[arm]/'motion_action_history.json').read_text());assert len(h)==a.updates and all(len(x['indices'])==16 and all(0<=i<2880 for i in x['indices']) for x in h)
for file,key in [('motion_action_center_history.json','indices'),('motion_action_history.json','indices'),('static_step_guard_history.json','static_indices'),('native_replay_history.json','indices'),('native_h32_history.json','indices')]:
 histories=[json.loads((paths[k]/file).read_text()) for k in ('control','candidate')];assert [[x[key] for x in h] for h in histories][0]==[[x[key] for x in h] for h in histories][1]
assert plans[0]['motion_action_center_enabled'] is False and plans[1]['motion_action_center_enabled'] is True
out={'passed':True,'updates':a.updates,'control_weights_optimizer_bitexact':True,'frozen_tensors':22,'adam_states_verified':8,'paired_indices_equal':True,'single_variable_verified':True,'points_per_update':2144,'policy_acceptance':False,'checkpoint_sha256':{k:hashlib.sha256((p/'world_model_calibrated.pt').read_bytes()).hexdigest() for k,p in paths.items()}};(a.root/'fit_integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
