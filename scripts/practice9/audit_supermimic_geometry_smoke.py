"""Verify trained geometry smoke artifacts and exact legacy training regression."""
import argparse,json
from pathlib import Path
import torch
from supermimic_geometry_model import GEOMETRY_KIND,load_geometry_artifact,configure_offline_geometry
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;s=r/'sole_geometry_fit_v1'
parent=torch.load(r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
rows=[]
for arm,enabled in [('control',False),('candidate',True)]:
 c=torch.load(s/f'smoke/{arm}/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
 assert c['kind']==GEOMETRY_KIND and c['calibration_updates']==4 and c['smoke_only']
 assert c['geometry_contract']['enabled']==enabled
 assert c['calibration_plan']['prediction_endpoints_per_update']==768
 model,predictor=load_geometry_artifact(c)
 for key,value in c['world_model'].items():
  assert torch.isfinite(value).all()
  if not key.startswith('net.'):assert torch.equal(value,parent['world_model'][key]),key
 assert torch.count_nonzero(c['world_model']['net.0.weight'][:,:2])==0
 extra=float(c['world_model']['net.0.weight'][:,137:].norm())
 assert (extra>0)==enabled
 assert all(float(v['step'])==4 for v in c['optimizer']['state'].values())
 with torch.no_grad():
  import numpy as np
  with np.load(r/'static_perturb_v1/seed42_amp0.00.npz') as z:v=[torch.tensor(z[k][:32],device='cuda:0') for k in ['state','action','context']]
  assert torch.isfinite(model(*v)).all() and torch.isfinite(predictor(*v)).all()
 rows.append(dict(arm=arm,extra_weight_norm=extra,geometry_initial_error=c['calibration_plan']['geometry_initial_error']))
old=torch.load(r/'root_pose_response_fit_v1/smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
new=torch.load(s/'legacy_smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert old['world_model'].keys()==new['world_model'].keys()
assert all(torch.equal(v,new['world_model'][k]) for k,v in old['world_model'].items())
out=dict(passed=True,rows=rows,legacy_training_weights_bit_identical=True,scope='Four-update smoke and loading/regression only; not prediction or policy acceptance')
(s/'smoke_integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
