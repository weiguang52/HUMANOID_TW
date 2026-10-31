"""All fit-pose conditions: separate linear/angular loss, scaling and gradients."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import load_geometry_artifact,install_geometry_input
from supermimic_root_pose_response import RootPoseResponseLoss
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
paths=[('initial_parent',r/'standing_wm_calibration_v1/fit/world_model_calibrated.pt'),('geometry_coefficient1',r/'sole_geometry_pose_balance_fit_v1/control/fit/world_model_calibrated.pt'),('geometry_measured_coefficient',r/'sole_geometry_pose_balance_fit_v1/candidate/fit/world_model_calibrated.pt')]
ref=torch.load(paths[1][1],map_location='cpu',weights_only=False);reports=[]
for label,path in paths:
 c=torch.load(path,map_location='cpu',weights_only=False);cfg=c['cfg'];w=c['world_model']
 if label=='initial_parent':
  m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w)
  with torch.no_grad():
   physical=torch.zeros(1,73,device='cuda:0');reference=m.state_norm.normalize(physical)[0,:2]
   m.net[0].bias.add_(m.net[0].weight[:,:2]@reference);m.net[0].weight[:,:2].zero_()
   install_geometry_input(m,ref['geometry_contract'],cfg)
 else:m,_=load_geometry_artifact(c)
 m.eval();snapshot={k:v.detach().clone() for k,v in m.state_dict().items()}
 for v in m.parameters():v.requires_grad_(False)
 for v in m.net.parameters():v.requires_grad_(True)
 loss_source=RootPoseResponseLoss(r/'root_pose_branches_v1',cfg);d=loss_source.data
 pred=m(d['state'].reshape(-1,73),d['action'].reshape(-1,30),d['context'].reshape(-1,34)).reshape(48,4,73)
 physical_error=(loss_source.derivative(pred)-loss_source.target)*loss_source.input_scale[loss_source.axes,None]
 physical_target=loss_source.target*loss_source.input_scale[loss_source.axes,None]
 std=(m.state_norm.var[7:13]+m.state_norm.eps).sqrt();std=torch.where(m.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
 error=physical_error/std[None,:]
 losses={'linear':.5*error[:,:3].square().mean(),'angular':.5*error[:,3:].square().mean()}
 params=list(m.net.named_parameters());gradients={};groups={}
 for group,loss in losses.items():
  gs=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True,allow_unused=True);pieces=[];extra=0.
  for (name,v),g in zip(params,gs):
   g=torch.zeros_like(v) if g is None else g.detach().clone()
   if name=='0.weight':g[:,:2]=0;extra=float(g[:,137:].norm())
   assert torch.isfinite(g).all();pieces.append(g.reshape(-1))
  gradient=torch.cat(pieces);gradients[group]=gradient
  sl=slice(0,3) if group=='linear' else slice(3,6)
  groups[group]=dict(half_weighted_loss=float(loss),gradient_norm=float(gradient.norm()),geometry_column_gradient_norm=extra,
   output_std=std[sl].tolist(),native_response_rms=float(physical_target[:,sl].square().mean().sqrt()),physical_response_error_rms=float(physical_error[:,sl].square().mean().sqrt()))
 g,h=gradients.values();den=float(g.norm()*h.norm());cosine=float(torch.dot(g,h))/den if den else None
 axes=[]
 for axis in range(6):
  selected=loss_source.axes==axis
  axes.append(dict(axis=axis,conditions=int(selected.sum()),normalized_linear_mse=float(error[selected,:3].square().mean()),normalized_angular_mse=float(error[selected,3:].square().mean())))
 for k,v in m.state_dict().items():assert torch.equal(v,snapshot[k]),k
 reports.append(dict(label=label,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),groups=groups,linear_angular_gradient_cosine=cosine,per_input_axis=axes,fit_files=loss_source.files,weights_unchanged=True))
 print(json.dumps({k:v for k,v in reports[-1].items() if k not in ['fit_files','per_input_axis']}),flush=True)
 del m,snapshot,gradients,losses,pred,g,h,gs,pieces,gradient
out=dict(reports=reports,fit_conditions=48,scope='Fit-only full 48 conditions. Half-weighted groups reproduce existing six-component mean loss. Parent is initialized with fixed XY invariance and zero geometry columns. No validation, optimizer update, loss reweighting or causal/control acceptance claim.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
