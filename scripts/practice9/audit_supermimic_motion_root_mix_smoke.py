"""Fit-only motion H1 source binding, frozen tensors and control regression."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',default='smoke');a=p.parse_args();r=a.root;s=r/'motion_root_mix_fit_v1'
parent=torch.load(r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
collection=r/'motion_pd_collection_cache_v2/full'
source_rows=json.loads((collection/'results.json').read_text())['rows']
selected=[x for x in source_rows if x['split']=='fit'];assert len(selected)==16
expected=[dict(file=x['trace'],sha256=hashlib.sha256((collection/x['trace']).read_bytes()).hexdigest(),states=x['steps'],source_family=x['source_family']) for x in selected]
assert sum(x['states'] for x in expected)==828 and sum(max(0,x['states']-7) for x in expected)==716
models={};rows=[]
for arm,flag in [('control',0),('candidate',1)]:
 c=torch.load(s/f'{a.stage}/{arm}/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False);plan=c['calibration_plan']
 assert c['calibration_updates']==4 and c['smoke_only']
 from supermimic_yaw_canonical import KIND,coordinate_metadata,apply_coordinate_contract,YawCanonicalModel
 assert c['kind']==KIND and plan['yaw_canonical'] is True and plan['mixed_horizon'] is False and plan['motion_h1_mix'] is True and plan['motion_root_mix'] is bool(flag)
 assert c['coordinate_contract']==plan['coordinate_contract']==coordinate_metadata(True)
 assert plan['motion_h8_mix']==1 and plan['motion_generator_seed']==48 and 'dynamic_h1_mix' not in plan
 h1meta=plan['motion_h1_input'];assert h1meta['files']==expected and h1meta['windows']==h1meta['states']==828 and not h1meta['validation_arrays_loaded']
 assert h1meta==json.loads((s/f'{a.stage}/{arm}/fit/motion_h1_input.json').read_text())
 assert plan['motion_h1_generator']==51
 meta=plan['motion_input'];assert meta['files']==expected and meta['windows']==716 and meta['states']==828 and not meta['validation_arrays_loaded']
 assert meta==json.loads((s/f'{a.stage}/{arm}/fit/motion_input.json').read_text())
 info=json.loads((collection/'input.json').read_text());assert info['asset_sha256']==c['cfg']['extra']['tw_mjcf_sha256'] and info['physics']==c['cfg']['extra']['tw_physics_contract']
 assert meta['audit_sha256']==hashlib.sha256((collection.parent/'full_audit.json').read_bytes()).hexdigest()
 long_meta=plan['long_motion_input'];assert long_meta['files']==expected and long_meta['windows']==337 and long_meta['window']==32
 assert long_meta['eligible_episodes']==14 and len(long_meta['short_episodes_retained_in_h8'])==2
 assert plan['actual_prediction_points_min']==plan['actual_prediction_points_max']==768 and plan['actual_prediction_points_total']==3072
 assert plan['long_static_windows']==3057 and plan['long_generators']==[49,50]
 assert plan['trajectory_batch']==64 and plan['long_trajectory_batch']==0
 assert plan['trajectory_batch']*8+plan['long_trajectory_batch']*32==512 and plan['rollout_endpoints']==512
 assert long_meta['files']==meta['files'] and not long_meta['validation_arrays_loaded']
 assert plan['prediction_endpoints_per_update']==768 and c['geometry_contract']['contract']=='sole_bottom_height_penetration_mm_v1'
 for k,v in c['world_model'].items():
  assert torch.isfinite(v).all()
  if not k.startswith('net.'):assert torch.equal(v,parent['world_model'][k]),k
 assert torch.count_nonzero(c['world_model']['net.0.weight'][:,:2])==0
 assert all(float(x['step'])==4 for x in c['optimizer']['state'].values())
 m,predictor=load_geometry_artifact(c)
 import numpy as np
 with np.load(collection/expected[0]['file']) as z:batch=[torch.tensor(z[k][:32],device='cuda:0') for k in ['state','action','context']]
 with torch.no_grad():
  from supermimic_pose_integration_diagnostic import PoseVelocityModel,apply_offline_prediction_contract
  actual=predictor(*batch);manual=PoseVelocityModel(YawCanonicalModel(m,True),c['prediction_contract'])(*batch)
  assert torch.isfinite(actual).all();torch.testing.assert_close(actual,manual,rtol=0,atol=1e-5)
  raw=apply_coordinate_contract(m,c)(*batch)
  torch.testing.assert_close(actual[:,7:13],raw[:,7:13],rtol=0,atol=1e-5)
  torch.testing.assert_close(actual[:,43:73],raw[:,43:73],rtol=0,atol=1e-5)
 invalid=[]
 bad=dict(c);bad.pop('coordinate_contract');invalid.append(bad)
 bad=dict(c);bad.pop('coordinate_contract');bad['calibration_plan']={k:v for k,v in plan.items() if k not in ['coordinate_contract','yaw_canonical']};invalid.append(bad)
 bad=dict(c);meta=dict(c['coordinate_contract'],contract='unknown');bad['coordinate_contract']=meta;bad['calibration_plan']=dict(plan,coordinate_contract=meta);invalid.append(bad)
 bad=dict(c);bad['calibration_plan']=dict(plan,yaw_canonical=False);invalid.append(bad)
 bad=dict(c,kind='offline_sole_geometry_dynamics_not_policy_checkpoint');invalid.append(bad)
 bad=dict(c);bad.pop('prediction_contract');invalid.append(bad)
 for bad in invalid:
  try:apply_offline_prediction_contract(m,bad)
  except ValueError:pass
  else:raise AssertionError('Invalid coordinate metadata accepted')
 from types import SimpleNamespace
 from supermimic.algo.trainer import SuperMimicTrainer
 try:SuperMimicTrainer.load(SimpleNamespace(device='cpu'),s/f'{a.stage}/{arm}/fit/world_model_calibrated.pt')
 except ValueError as error:assert 'Offline calibration artifacts' in str(error)
 else:raise AssertionError('Offline yaw artifact accepted as policy')
 from supermimic_motion_root_response import MotionRootResponseLoss,load_motion_root_response,derivative
 root=Path(plan['motion_root_collection']);source_rows=json.loads((root/'results.json').read_text())['rows']
 forbidden={(root/x['file']).resolve() for x in source_rows if x['split']=='validation'}
 original_load=np.load;opened=[]
 def monitored_load(path,*args,**kwargs):
  resolved=Path(path).resolve();assert resolved not in forbidden
  opened.append(resolved);return original_load(path,*args,**kwargs)
 np.load=monitored_load
 try:probe=MotionRootResponseLoss(root,c['cfg'])
 finally:np.load=original_load
 assert len(opened)==192 and probe.meta==plan['motion_root_input']
 assert not probe.meta['validation_arrays_loaded'] and probe.meta['conditions']==576
 assert probe.meta==json.loads((s/f'{a.stage}/{arm}/fit/motion_root_input.json').read_text())
 assert plan['motion_root_generator']==52
 from supermimic_yaw_canonical import apply_coordinate_contract
 rm=apply_coordinate_contract(m,c);idx=probe.sample();points={'n':0}
 def tally(module,args):points['n']+=len(args[0])
 handle=m.register_forward_pre_hook(tally)
 with torch.no_grad():actual_loss=probe(rm,idx)
 handle.remove();assert points['n']==32
 with torch.no_grad():
  values=rm(*[probe.data[k][idx].reshape(-1,width) for k,width in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  response=(values[:,1,7:13]-values[:,0,7:13]+values[:,2,7:13]-values[:,3,7:13])/probe.den[idx]
  target=(probe.data['next_state'][idx,1,7:13]-probe.data['next_state'][idx,0,7:13]+probe.data['next_state'][idx,2,7:13]-probe.data['next_state'][idx,3,7:13])/probe.den[idx]
  std=(m.state_norm.var[7:13]+m.state_norm.eps).sqrt();std=torch.where(m.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
  manual_loss=((response-target)*std[probe.axes[idx],None]/std[None,:]).square().mean()
  torch.testing.assert_close(actual_loss,manual_loss,rtol=0,atol=0)
 models[arm]=c;rows.append(dict(arm=arm,yaw_enabled=True,motion_root_mix=bool(flag),motion_h8_mix=1,fit_motion_files=16,fit_motion_states=828,fit_motion_h8_windows=716))
old=torch.load(r/'motion_h1_anchor_fit_v1/smoke/candidate/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,old['world_model'][k]) for k,v in models['control']['world_model'].items())
changed=[k for k,v in models['candidate']['world_model'].items() if not torch.equal(v,models['control']['world_model'][k])]
assert set(changed)=={'net.'+str(i)+'.'+suffix for i in [0,2,4,6] for suffix in ['weight','bias']}
out=dict(passed=True,negative_contract_cases_per_arm=6,policy_loader_rejected_both=True,shared_and_manual_predictors_match=True,response_velocity_matches_full_predictor=True,rows=rows,legacy_control_weights_bit_identical=True,candidate_changed_tensors=changed,scope='4-update source/loader/freeze regression only; no held-out accuracy or policy acceptance')
(s/(a.stage+'_integrity.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
