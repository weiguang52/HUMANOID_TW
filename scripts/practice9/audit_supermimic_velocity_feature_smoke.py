"""Velocity-feature smoke invariants and legacy height-path training regression."""
import argparse,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact,GEOMETRY_VELOCITY_CONTRACT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;s=r/'sole_velocity_feature_fit_v1'
parent=torch.load(r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False);rows=[]
for arm,enabled in [('control',False),('candidate',True)]:
 c=torch.load(s/f'smoke/{arm}/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
 initial=torch.load(r/'sole_geometry_velocity_contract_v2'/(arm+'_initial.pt'),map_location='cpu',weights_only=False)
 assert c['calibration_updates']==4 and c['smoke_only'] and c['calibration_plan']['prediction_endpoints_per_update']==768
 meta=c['geometry_contract'];assert meta==c['calibration_plan']['geometry_contract']
 assert meta['contract']==GEOMETRY_VELOCITY_CONTRACT and meta['enabled'] and meta['velocity_enabled']==enabled
 assert meta['xy_reference_physical']==[0.,0.]
 for k,v in c['world_model'].items():
  assert torch.isfinite(v).all()
  if not k.startswith('net.'):assert torch.equal(v,parent['world_model'][k]),k
  if k.startswith('net.0.') and k not in ['net.0.weight','net.0.bias']:assert torch.equal(v,initial['world_model'][k]),k
 weight=c['world_model']['net.0.weight'];assert weight.shape[1]==161 and torch.count_nonzero(weight[:,:2])==0
 velocity_norm=float(weight[:,153:].norm());assert (velocity_norm>0)==enabled
 assert all(float(x['step'])==4 for x in c['optimizer']['state'].values())
 m,predictor=load_geometry_artifact(c)
 import numpy as np
 with np.load(r/'static_perturb_v1/seed42_amp0.00.npz') as z:v=[torch.tensor(z[k][:32],device='cuda:0') for k in ['state','action','context']]
 with torch.no_grad():assert torch.isfinite(m(*v)).all() and torch.isfinite(predictor(*v)).all()
 rows.append(dict(arm=arm,velocity_columns_norm=velocity_norm,initial_prediction_error=c['calibration_plan']['geometry_initial_error']))
old=torch.load(r/'sole_geometry_fit_v1/smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
new=torch.load(s/'legacy_smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert old['world_model'].keys()==new['world_model'].keys()
assert all(torch.equal(v,new['world_model'][k]) for k,v in old['world_model'].items())
out=dict(passed=True,rows=rows,legacy_height_path_weights_bit_identical=True,scope='4-update training/loader invariants only; no accuracy or control acceptance')
(s/'smoke_integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
