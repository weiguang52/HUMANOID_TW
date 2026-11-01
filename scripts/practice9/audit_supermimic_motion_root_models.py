"""Frozen native motion-root response evaluation; never an optimizer."""
import argparse,hashlib,json
from pathlib import Path
import torch
from supermimic_geometry_model import configure_offline_geometry
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
from supermimic_motion_root_response import load_motion_root_response,derivative
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;reports=[];evidence=[]
for arm,relative in [('source','standing_wm_calibration_v1/fit/world_model_calibrated.pt'),('control','motion_h1_anchor_fit_v1/control/fit/world_model_calibrated.pt'),('candidate','motion_h1_anchor_fit_v1/candidate/fit/world_model_calibrated.pt')]:
 path=r/relative;c=torch.load(path,map_location='cpu',weights_only=False);cfg=c['cfg'];w=c['world_model']
 m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
 configure_offline_geometry(m,c);m.load_state_dict(w,strict=True);m.eval();predict=apply_offline_prediction_contract(m,c)
 for v in m.parameters():v.requires_grad_(False)
 rows=[]
 for split in ['fit','validation']:
  data,den,axes,meta=load_motion_root_response(r/'motion_root_velocity_branches_v1/serial_full',c['cfg'],split)
  preds=[]
  with torch.no_grad():
   for start in range(0,576,32):preds.append(predict(*[data[k][start:start+32].reshape(-1,width) for k,width in [('state',73),('action',30),('context',34)]]).reshape(-1,4,73))
   pred=torch.cat(preds);assert torch.isfinite(pred).all();response=derivative(pred,den);target=derivative(data['next_state'],den)
   std=(m.state_norm.var[7:13]+m.state_norm.eps).sqrt();std=torch.where(m.state_norm.frozen_mask[7:13],torch.ones_like(std),std)
   normalized=(response-target)*std[axes,None]/std[None,:]
   err=pred-data['next_state'];rows.append(dict(split=split,conditions=576,absolute_velocity_rmse={k:float(err[:,:,sl].square().mean().sqrt()) for k,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint_velocity',slice(43,73))]},response_normalized_mse=float(normalized.square().mean()),response_rmse_per_output=(response-target).square().mean(0).sqrt().tolist(),native_response_frobenius_mean=float(target.square().sum(1).mean().sqrt())))
  evidence.append(meta)
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
 report=dict(arm=arm,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows);reports.append(report);print(json.dumps(report),flush=True)
 del m,predict,c
out=dict(models=reports,evidence=evidence,weights_unchanged=True,optimizer_updates=0,scope='All fixed source-isolated motion-root conditions fit/validation, evaluation only. Root normalized derivative metric uses immutable source normalization, units mixed by axis. Finite branches near contact are not infinitesimal global Jacobians or policy stability acceptance.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
