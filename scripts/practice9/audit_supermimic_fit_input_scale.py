"""Fixed fit-only input ranges after actual shared predictor canonicalization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
from supermimic_motion_root_response import load_motion_root_response
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
path=r/'motion_root_mix_fit_v1/control/fit/world_model_calibrated.pt';c=torch.load(path,map_location='cpu',weights_only=False);m,predict=load_geometry_artifact(c);m.eval();mapping=json.loads((r/'motion_pd_collection_cache_v2/full/control_mapping.json').read_text())
keys=['state','action','context'];fit=json.loads((r/'motion_root_mix_fit_v1/control/fit/fit_input.json').read_text());sources=[];datasets={}
for label,directory,rows in [('static_trajectory','static_perturb_v1',fit['trajectory_files']),('static_branch','orthogonal_branches_v1',fit['one_step_files'])]:
 data={k:[] for k in keys}
 for row in rows:
  p=r/directory/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
  with np.load(p,allow_pickle=False) as z:
   for k in keys:data[k].append(z[k].copy())
  sources.append(dict(file=str(p),sha256=row['sha256']))
 datasets[label]={k:torch.tensor(np.concatenate(v),device='cuda:0') for k,v in data.items()}
motion,meta=load_motion_fit_windows(r/'motion_pd_collection_cache_v2/full',r/'motion_pd_collection_cache_v2/full_audit.json',mapping,window=1)
datasets['motion']={k:(motion[k] if k=='state' else motion[k][:,0]) for k in keys}
branch,_,_,bmeta=load_motion_root_response(r/'motion_root_velocity_branches_v1/serial_full',c['cfg'])
datasets['motion_root_branch']={k:branch[k].reshape(-1,branch[k].shape[-1]) for k in keys}
captured=[]
def hook(module,args):captured.append(args[0].detach().cpu())
handle=m.net[0].register_forward_pre_hook(hook);reports=[]
groups={'root_z':slice(2,3),'quaternion':slice(3,7),'root_linear_velocity':slice(7,10),'root_angular_velocity':slice(10,13),'joint_position':slice(13,43),'joint_velocity':slice(43,73),'action':slice(73,103),'context':slice(103,137)}
with torch.no_grad():
 for label,data in datasets.items():
  captured.clear()
  for start in range(0,len(data['state']),128):predict(*[data[k][start:start+128] for k in keys])
  x=torch.cat(captured);assert x.shape==(len(data['state']),137) and torch.isfinite(x).all()
  reports.append(dict(dataset=label,states=len(x),groups={k:dict(abs_p95=float(torch.quantile(x[:,sl].abs().flatten(),.95)),abs_max=float(x[:,sl].abs().max())) for k,sl in groups.items()},per_feature_abs_p95=torch.quantile(x.abs(),.95,dim=0).tolist(),per_feature_abs_max=x.abs().max(0).values.tolist()))
handle.remove()
for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
out=dict(reports=reports,sources=sources,motion_input=meta,motion_root_input=bmeta,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),weights_unchanged=True,validation_arrays_loaded=False,scope='Actual first-layer137 normalized input, after shared yaw canonicalization and before appended geometry features. Fit only, no optimizer or new normalization. Ranges alone do not prove ill-conditioning or a successful fix.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(reports))
