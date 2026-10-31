"""Fit-only motion H8 source binding, frozen tensors and control regression."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;s=r/'yaw_canonical_fit_v1'
parent=torch.load(r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
collection=r/'motion_pd_collection_cache_v2/full'
source_rows=json.loads((collection/'results.json').read_text())['rows']
selected=[x for x in source_rows if x['split']=='fit'];assert len(selected)==16
expected=[dict(file=x['trace'],sha256=hashlib.sha256((collection/x['trace']).read_bytes()).hexdigest(),states=x['steps'],source_family=x['source_family']) for x in selected]
assert sum(x['states'] for x in expected)==828 and sum(max(0,x['states']-7) for x in expected)==716
models={};rows=[]
for arm,flag in [('control',0),('candidate',1)]:
 c=torch.load(s/f'smoke/{arm}/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False);plan=c['calibration_plan']
 assert c['calibration_updates']==4 and c['smoke_only']
 from supermimic_yaw_canonical import KIND,coordinate_metadata,apply_coordinate_contract,YawCanonicalModel
 assert c['kind']==KIND and plan['yaw_canonical'] is bool(flag)
 assert c['coordinate_contract']==plan['coordinate_contract']==coordinate_metadata(flag)
 assert plan['motion_h8_mix']==1 and plan['motion_generator_seed']==48 and 'dynamic_h1_mix' not in plan
 meta=plan['motion_input'];assert meta['files']==expected and meta['windows']==716 and meta['states']==828 and not meta['validation_arrays_loaded']
 assert meta==json.loads((s/f'smoke/{arm}/fit/motion_input.json').read_text())
 info=json.loads((collection/'input.json').read_text());assert info['asset_sha256']==c['cfg']['extra']['tw_mjcf_sha256'] and info['physics']==c['cfg']['extra']['tw_physics_contract']
 assert meta['audit_sha256']==hashlib.sha256((collection.parent/'full_audit.json').read_bytes()).hexdigest()
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
  actual=predictor(*batch);manual=PoseVelocityModel(YawCanonicalModel(m,bool(flag)),c['prediction_contract'])(*batch)
  assert torch.isfinite(actual).all();torch.testing.assert_close(actual,manual,rtol=0,atol=1e-5)
  raw=apply_coordinate_contract(m,c)(*batch)
  torch.testing.assert_close(actual[:,7:13],raw[:,7:13],rtol=0,atol=1e-5)
  torch.testing.assert_close(actual[:,43:73],raw[:,43:73],rtol=0,atol=1e-5)
 invalid=[]
 bad=dict(c);bad.pop('coordinate_contract');invalid.append(bad)
 bad=dict(c);bad.pop('coordinate_contract');bad['calibration_plan']={k:v for k,v in plan.items() if k not in ['coordinate_contract','yaw_canonical']};invalid.append(bad)
 bad=dict(c);meta=dict(c['coordinate_contract'],contract='unknown');bad['coordinate_contract']=meta;bad['calibration_plan']=dict(plan,coordinate_contract=meta);invalid.append(bad)
 bad=dict(c);bad['calibration_plan']=dict(plan,yaw_canonical=not bool(flag));invalid.append(bad)
 bad=dict(c,kind='offline_sole_geometry_dynamics_not_policy_checkpoint');invalid.append(bad)
 bad=dict(c);bad.pop('prediction_contract');invalid.append(bad)
 for bad in invalid:
  try:apply_offline_prediction_contract(m,bad)
  except ValueError:pass
  else:raise AssertionError('Invalid coordinate metadata accepted')
 from types import SimpleNamespace
 from supermimic.algo.trainer import SuperMimicTrainer
 try:SuperMimicTrainer.load(SimpleNamespace(device='cpu'),s/f'smoke/{arm}/fit/world_model_calibrated.pt')
 except ValueError as error:assert 'Offline calibration artifacts' in str(error)
 else:raise AssertionError('Offline yaw artifact accepted as policy')
 models[arm]=c;rows.append(dict(arm=arm,yaw_enabled=bool(flag),motion_h8_mix=1,fit_motion_files=16,fit_motion_states=828,fit_motion_h8_windows=716))
old=torch.load(r/'motion_h8_mixture_fit_v1/smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,old['world_model'][k]) for k,v in models['control']['world_model'].items())
changed=[k for k,v in models['candidate']['world_model'].items() if not torch.equal(v,models['control']['world_model'][k])]
assert set(changed)=={'net.'+str(i)+'.'+suffix for i in [0,2,4,6] for suffix in ['weight','bias']}
out=dict(passed=True,negative_contract_cases_per_arm=6,policy_loader_rejected_both=True,shared_and_manual_predictors_match=True,response_velocity_matches_full_predictor=True,rows=rows,legacy_control_weights_bit_identical=True,candidate_changed_tensors=changed,scope='4-update source/loader/freeze regression only; no held-out accuracy or policy acceptance')
(s/'smoke_integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
