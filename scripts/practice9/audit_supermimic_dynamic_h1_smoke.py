"""Fit-only dynamic H1 source binding, frozen tensors and control regression."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;s=r/'dynamic_h1_mixture_fit_v1'
parent=torch.load(r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
source_rows=json.loads((r/'static_perturb_v1/results.json').read_text())['rows']
selected=[x for x in source_rows if x['split']=='fit' and x['mode']!='amplitude_0.00']
expected=[dict(file=x['trace'],sha256=hashlib.sha256((r/'static_perturb_v1'/x['trace']).read_bytes()).hexdigest()) for x in selected]
assert len(expected)==6;models={};rows=[]
for arm,flag in [('control',0),('candidate',1)]:
 c=torch.load(s/f'smoke/{arm}/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False);plan=c['calibration_plan']
 assert c['calibration_updates']==4 and c['smoke_only']
 assert plan['dynamic_h1_mix']==flag and plan['dynamic_h1_count']==1305 and plan['dynamic_h1_files']==expected and plan['dynamic_h1_generator_seed']==47
 assert plan['prediction_endpoints_per_update']==768 and c['geometry_contract']['contract']=='sole_bottom_height_penetration_mm_v1'
 for k,v in c['world_model'].items():
  assert torch.isfinite(v).all()
  if not k.startswith('net.'):assert torch.equal(v,parent['world_model'][k]),k
 assert torch.count_nonzero(c['world_model']['net.0.weight'][:,:2])==0
 assert all(float(x['step'])==4 for x in c['optimizer']['state'].values())
 m,predictor=load_geometry_artifact(c)
 import numpy as np
 with np.load(r/'static_perturb_v1'/expected[0]['file']) as z:batch=[torch.tensor(z[k][:32],device='cuda:0') for k in ['state','action','context']]
 with torch.no_grad():assert torch.isfinite(m(*batch)).all() and torch.isfinite(predictor(*batch)).all()
 models[arm]=c;rows.append(dict(arm=arm,mix=flag,fit_dynamic_files=6,fit_dynamic_states=1305))
old=torch.load(r/'sole_geometry_fit_v1/smoke/candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,old['world_model'][k]) for k,v in models['control']['world_model'].items())
changed=[k for k,v in models['candidate']['world_model'].items() if not torch.equal(v,models['control']['world_model'][k])]
assert changed and all(k.startswith('net.') for k in changed)
out=dict(passed=True,rows=rows,legacy_control_weights_bit_identical=True,candidate_changed_tensors=changed,scope='4-update source/loader/freeze regression only; no held-out accuracy or policy acceptance')
(s/'smoke_integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
