"""Fit-only fixed-batch gradient conflicts; no optimizer or validation access."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact,install_geometry_input
from supermimic_antithetic_response import AntitheticResponseLoss
from supermimic_root_state_response import RootStateResponseLoss
from supermimic_root_pose_response import RootPoseResponseLoss
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
pair=r/'sole_geometry_fit_v1';fit=json.loads((pair/'control/fit/fit_input.json').read_text());sources=[]
keys=['state','action','context','next_state'];trajectories=[];one=[]
for group,dirname,rows in [('trajectory','static_perturb_v1',fit['trajectory_files']),('branch','orthogonal_branches_v1',fit['one_step_files'])]:
 for row in rows:
  path=(r/dirname/row['file']).resolve();assert path.is_relative_to((r/dirname).resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in keys}
  (trajectories if group=='trajectory' else one).append(d)
  sources.append(dict(group=group,file=str(path),sha256=row['sha256']))
traj={k:[] for k in keys}
for d in trajectories:
 assert np.array_equal(d['next_state'][:-1],d['state'][1:])
 for start in range(len(d['state'])-7):
  traj['state'].append(d['state'][start])
  for k in keys[1:]:traj[k].append(d[k][start:start+8])
traj={k:torch.tensor(np.stack(v),device='cuda:0') for k,v in traj.items()}
one={k:torch.tensor(np.concatenate([d[k] for d in one]),device='cuda:0') for k in keys}
assert len(traj['state'])==3249 and len(one['state'])==1200
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import PoseVelocityModel
source=r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt'
c=torch.load(source,map_location='cpu',weights_only=False);cfg=c['cfg'];w=c['world_model']
reference=torch.load(pair/'candidate/fit/world_model_calibrated.pt',map_location='cpu',weights_only=False)
assert reference['geometry_contract']['enabled']
m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w)
with torch.no_grad():
 query=traj['state'][:64].clone();query[:,:2]=0
 qa=traj['action'][:64,0];qc=traj['context'][:64,0]
 for _ in range(5):m(query,qa,qc)
 before=m(query,qa,qc)
 physical_xy=m.state_norm.normalize(query)[0,:2]
 m.net[0].bias.add_(m.net[0].weight[:,:2]@physical_xy);m.net[0].weight[:,:2].zero_()
 install_geometry_input(m,reference['geometry_contract'],cfg)
 for _ in range(5):m(query,qa,qc)
 initial_error=float((m(query,qa,qc)-before).abs().max());assert initial_error<=1e-5
predictor=PoseVelocityModel(m,reference['prediction_contract']);m.eval()
for v in m.parameters():v.requires_grad_(False)
for v in m.net.parameters():v.requires_grad_(True)
params=list(m.net.named_parameters());snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
primary=[torch.zeros_like(v,dtype=torch.float64) for _,v in params]
def accumulate(loss,buffers):
 gradients=torch.autograd.grad(loss,[v for _,v in params],allow_unused=True)
 for (_,p),g,b in zip(params,gradients,buffers):
  if g is not None:
   assert torch.isfinite(g).all();b.add_(g.detach().double())
base_value=0.
for start in range(0,len(traj['state']),64):
 end=min(start+64,len(traj['state']));pred=traj['state'][start:end];loss=0
 for j in range(8):
  pred=predictor(pred,traj['action'][start:end,j],traj['context'][start:end,j])
  loss+=m.state_loss(pred,traj['next_state'][start:end,j])/8
 weighted=.5*(end-start)/len(traj['state'])*loss
 base_value+=float(weighted);accumulate(weighted,primary)
 if start%512==0:print('H8 windows measured',end,flush=True)
for start in range(0,len(one['state']),64):
 end=min(start+64,len(one['state']))
 pred=predictor(one['state'][start:end],one['action'][start:end],one['context'][start:end])
 loss=.5*(end-start)/len(one['state'])*m.state_loss(pred,one['next_state'][start:end])
 base_value+=float(loss);accumulate(loss,primary)
pose=RootPoseResponseLoss(r/'root_pose_branches_v1',cfg);d=pose.data
pred=m(d['state'].reshape(-1,73),d['action'].reshape(-1,30),d['context'].reshape(-1,34)).reshape(48,4,73)
response=pose.derivative(pred);std=(m.state_norm.var[7:13]+m.state_norm.eps).sqrt()
std=torch.where(m.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
error=(response-pose.target)*pose.input_scale[pose.axes,None]/std[None,:]
pose_value=error.square().mean();pose_grad=[torch.zeros_like(v,dtype=torch.float64) for _,v in params]
accumulate(pose_value,pose_grad)
for buffers in [primary,pose_grad]:
 assert params[0][0]=='0.weight';buffers[0][:,:2]=0
norm=lambda v:float(torch.sqrt(sum(x.square().sum() for x in v)))
base_norm=norm(primary);pose_norm=norm(pose_grad);assert base_norm>0 and pose_norm>0
coefficient=base_norm/pose_norm
assert np.isfinite(coefficient) and coefficient>0
for key,v in m.state_dict().items():assert torch.equal(v,snapshot[key]),key
out=dict(coefficient=coefficient,primary_gradient_norm=base_norm,pose_gradient_norm=pose_norm,primary_loss=base_value,pose_loss=float(pose_value),
 primary_h8_windows=3249,primary_h1_states=1200,pose_conditions=48,source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 initialization_geometry_contract=reference['geometry_contract'],initial_prediction_error=initial_error,prediction_contract=reference['prediction_contract'],
 sources=sources,pose_fit_files=pose.files,weights_unchanged=True,new_optimizer_updates=0,
 rule='One frozen coefficient: full-fit primary (.5H8+.5H1) projected gradient norm divided by full-fit pose-response gradient norm; no validation or search')
(a.output/'measurement.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['sources','pose_fit_files','prediction_contract','initialization_geometry_contract']}),flush=True)
