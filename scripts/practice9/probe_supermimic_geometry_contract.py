"""Initial geometry model equivalence, serialization, gradients and policy isolation."""
import argparse,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic_geometry_model import GEOMETRY_KIND,GEOMETRY_CONTRACT,install_geometry_input,load_geometry_artifact
p=argparse.ArgumentParser(description=__doc__)
for k in ['root','asset','output']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;source=r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt'
c=torch.load(source,map_location='cpu',weights_only=False);cfg=c['cfg'];w=c['world_model']
base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();base.load_state_dict(w)
fit=json.loads((r/'root_pose_response_fit_v1/control/fit/fit_input.json').read_text())
inputs=[]
for directory,key in [('static_perturb_v1','trajectory_files'),('orthogonal_branches_v1','one_step_files')]:
 for row in fit[key]:
  path=(r/directory/row['file']).resolve();assert path.is_relative_to((r/directory).resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:inputs.append([torch.tensor(z[k],device='cuda:0') for k in ['state','action','context']])
contract=json.loads((r/'root_pose_response_fit_v1/control/fit/world_model_calibrated.pt').with_name('plan.json').read_text())['prediction_contract']
reports=[]
for enabled in [False,True]:
 model=copy.deepcopy(base);metadata=dict(contract=GEOMETRY_CONTRACT,enabled=enabled,height_scale_m=.001,ground_height_m=0,asset=str(a.asset),asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest())
 install_geometry_input(model,metadata,cfg);maximum=0.;feature_error=0.;frames=0
 with torch.no_grad():
  for values in inputs:
   for start in range(0,len(values[0]),256):
    v=[x[start:start+256] for x in values]
    x=torch.cat([model.state_norm.normalize(v[0]),model.action_norm.normalize(v[1]),model.context_norm.normalize(v[2])],-1)
    h=model.net[0].geometry(v[0])[...,[0,2,4,6]].reshape(len(v[0]),8)/.001
    direct=torch.cat([h,torch.relu(-h)],-1)
    feature_error=max(feature_error,float((model.net[0].geometry_features(x)-direct).abs().max()))
    for _ in range(5):base(*v);model(*v)
    maximum=max(maximum,float((base(*v)-model(*v)).abs().max()));frames+=len(v[0])
 assert frames==4505 and maximum<=1e-5,(frames,maximum)
 assert feature_error*.001<=1e-6,feature_error
 artifact=dict(kind=GEOMETRY_KIND,world_model=model.state_dict(),cfg=cfg,geometry_contract=metadata,prediction_contract=contract,calibration_updates=0,scope='Untrained contract probe only')
 file=a.output/('candidate_initial.pt' if enabled else 'control_initial.pt');torch.save(artifact,file)
 loaded,predictor=load_geometry_artifact(torch.load(file,map_location='cpu',weights_only=False))
 v=[x[:64] for x in inputs[0]]
 with torch.no_grad():roundtrip=float((loaded(*v)-model(*v)).abs().max())
 assert roundtrip<=1e-5
 for k,value in model.state_dict().items():
  if not k.startswith('net.'):assert torch.equal(value.cpu(),w[k]),k
 model.zero_grad();model(*v)[:,7:13].sum().backward()
 grad=model.net[0].weight.grad[:,137:];assert torch.isfinite(grad).all()
 if enabled:assert torch.count_nonzero(grad)>0
 else:assert torch.count_nonzero(grad)==0
 try:SuperMimicTrainer.load(SimpleNamespace(device='cpu'),file)
 except ValueError as error:assert str(error)=='Offline calibration artifacts cannot load as policy checkpoints'
 else:raise AssertionError('Policy loader failed to reject offline model')
 reports.append(dict(enabled=enabled,frames=frames,reconstructed_feature_error_m=feature_error*.001,initial_prediction_error=maximum,roundtrip_error=roundtrip,feature_weight_gradient_norm=float(grad.norm()),policy_loader_rejected_before_weights=True))
out=dict(passed=True,reports=reports,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Untrained contract/serialization proof, not held-out accuracy or control acceptance')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
