"""Fit-only function-preserving scale/export contract before optimization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
from supermimic_motion_root_response import load_motion_root_response
from supermimic_training_input_scale import TrainingInputScale,fit_scale,INDICES,RULE
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
source_path=r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt';source=torch.load(source_path,map_location='cpu',weights_only=False)
template=torch.load(r/'motion_root_mix_fit_v1/control/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
m,predict=load_geometry_artifact(template);m.eval();fresh=m.state_dict()
for k,v in source['world_model'].items():
 if k=='net.0.weight':v=torch.cat([v,torch.zeros(v.shape[0],16)],1)
 fresh[k]=v
m.load_state_dict(fresh)
with torch.no_grad():
 first=m.net[0];ref=-m.state_norm.mean[:2]/(m.state_norm.var[:2]+m.state_norm.eps).sqrt()
 first.bias.add_(first.weight[:,:2]@ref);first.weight[:,:2].zero_()
original={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
keys=['state','action','context'];fit=json.loads((r/'motion_root_mix_fit_v1/control/fit/fit_input.json').read_text());data={k:[] for k in keys};files=[]
for row in fit['trajectory_files']:
 path=r/'static_perturb_v1'/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 with np.load(path,allow_pickle=False) as z:
  for k in keys:data[k].append(z[k].copy())
 files.append(row)
static={k:torch.tensor(np.concatenate(v),device='cuda:0') for k,v in data.items()}
mapping=json.loads((r/'motion_pd_collection_cache_v2/full/control_mapping.json').read_text())
w,meta=load_motion_fit_windows(r/'motion_pd_collection_cache_v2/full',r/'motion_pd_collection_cache_v2/full_audit.json',mapping,window=1)
motion={k:(w[k] if k=='state' else w[k][:,0]) for k in keys}
branch,_,_,bmeta=load_motion_root_response(r/'motion_root_velocity_branches_v1/serial_full',source['cfg'])
query={k:torch.cat([static[k],motion[k],branch[k].reshape(-1,branch[k].shape[-1])]) for k in keys}
scale=fit_scale(m,static,motion);assert len(query['state'])==6437
baseline=[]
with torch.no_grad():
 for start in range(0,6437,128):baseline.append(predict(*[query[k][start:start+128] for k in keys]))
baseline=torch.cat(baseline)
rows=[]
for enabled in [False,True]:
 m.load_state_dict(original);transform=TrainingInputScale(m.net[0],scale,enabled)
 values=[]
 with torch.no_grad():
  for start in range(0,6437,128):values.append(predict(*[query[k][start:start+128] for k in keys]))
 actual=torch.cat(values);error=float((actual-baseline).abs().max());assert error<=1e-5
 if not enabled:assert torch.equal(actual,baseline)
 s=query['state'][::402].clone().requires_grad_(True);act=query['action'][::402].clone().requires_grad_(True);ctx=query['context'][::402]
 g=torch.autograd.grad(predict(s,act,ctx)[:,7:13].square().sum(),[s,act]);assert all(torch.isfinite(x).all() for x in g)
 # Export tested after a fixed non-optimizer parameter perturbation as well.
 with torch.no_grad():
  old_weight=m.net[0].weight[:,INDICES].clone()
  m.net[0].weight[:,INDICES]+=1e-6
  assert not torch.equal(m.net[0].weight[:,INDICES],old_weight)
 before=[]
 with torch.no_grad():
  for start in range(0,6437,128):before.append(predict(*[query[k][start:start+128] for k in keys]))
 before=torch.cat(before);transform.export();after=[]
 with torch.no_grad():
  for start in range(0,6437,128):after.append(predict(*[query[k][start:start+128] for k in keys]))
 export_error=float((torch.cat(after)-before).abs().max());assert export_error<=1e-5
 rows.append(dict(enabled=enabled,queries=6437,initial_max_error=error,export_max_error=export_error,finite_state_action_gradients=True,parameter_perturbation_verified=True))
m.load_state_dict(original)
for k,v in m.state_dict().items():assert torch.equal(v.cpu(),original[k]),k
out=dict(passed=True,rule=RULE,scale=scale.cpu().tolist(),indices=INDICES,static_states=3305,motion_states=828,static_files=files,motion_input=meta,motion_root_input=bmeta,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),rows=rows,optimizer_updates=0,validation_arrays_loaded=False,weights_restored=True,scope='Function-preserving first geometry-layer training coordinates only; fixed perturbation tests export, not a learned policy or efficacy claim. Geometry computed before scaling from unchanged original state input.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['static_files','motion_input','motion_root_input','scale','indices']}))
