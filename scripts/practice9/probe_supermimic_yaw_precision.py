"""Separate float32 rotated-observation rounding from mathematical yaw covariance."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
from supermimic_yaw_canonical import YawCanonicalModel
from supermimic_yaw_transform import rotate_world_state,state_metrics
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;source=r/'yaw_canonical_contract_probe_v1/results/predictions.npz'
with np.load(source,allow_pickle=False) as z:data={k:torch.tensor(z[k],device='cuda:0') for k in ['state','action','context']};prior=torch.tensor(z['motion_mix_canonical'],device='cuda:0')
path=r/'motion_h8_mixture_fit_v1/candidate/fit/world_model_calibrated.pt';artifact=torch.load(path,map_location='cpu',weights_only=False)
base,_=load_geometry_artifact(artifact);model=apply_offline_prediction_contract(YawCanonicalModel(base,True),artifact)
rows=[]
with torch.no_grad():
 for dtype in [torch.float32,torch.float64]:
  state=data['state'].to(dtype);pred=model(state,data['action'],data['context']);assert torch.isfinite(pred).all()
  if dtype==torch.float32:regression=state_metrics(pred,prior);assert regression['max_state_component_error']<=1e-5
  for turns in [0,1,2]:
   rotated=model(rotate_world_state(state,turns),data['action'],data['context'])
   metrics=state_metrics(rotated,rotate_world_state(pred,turns));rows.append(dict(input_dtype=str(dtype),yaw_degrees=turns*90,metrics=metrics,within_original_structure_atol=metrics['max_state_component_error']<=1e-5))
 for k,v in base.state_dict().items():assert torch.equal(v.cpu(),artifact['world_model'][k]),k
out=dict(rows=rows,float32_path_regression=regression,weights_unchanged=True,source_prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Precision attribution only. Original float32 structure failures remain failures; float64 inputs do not prove float32 robustness or physical accuracy. Network features remain float32, no fitting or original gate changes.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
