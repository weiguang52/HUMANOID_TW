"""Fit-only frozen joint-action Jacobian residual; diagnostic only, never export a policy."""
import argparse,hashlib,json,datetime,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_antithetic_response import AntitheticResponseLoss
from supermimic_geometry_model import configure_offline_geometry
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
from evaluate_supermimic_tw import audit_world_model
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--branches',type=Path,required=True);p.add_argument('--integrity',type=Path,required=True);p.add_argument('--collection',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg']
raw=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();raw=configure_offline_geometry(raw,c);raw.load_state_dict(w,strict=True);raw.eval()
model=apply_offline_prediction_contract(raw,c);targets=[x for x in model.modules() if hasattr(x,'net') and hasattr(x,'action_norm')];assert len(targets)==1;raw=targets[0];before={k:v.detach().clone() for k,v in raw.state_dict().items()};original_forward=raw.forward
loss=AntitheticResponseLoss(a.branches,a.integrity);d=loss.data
with torch.no_grad():
 pred=model(d['state'].reshape(-1,73),d['action'].reshape(-1,30),d['context'].reshape(-1,34)).reshape(240,4,73)
 response=loss.derivative(pred);residual=(loss.target[:,43:73]-response[:,43:73]).double()
 design=loss.derivative(d['action']).double();assert torch.equal(design,torch.eye(30,device='cuda').repeat(8,1))
 coefficient=torch.linalg.lstsq(design,residual).solution
 assert coefficient.shape==(30,30) and torch.isfinite(coefficient).all()
 def corrected(state,action,context):
  value=original_forward(state,action,context);delta=(action.to(coefficient.dtype)@coefficient).to(value.dtype)
  result=torch.cat([value[:,:43],value[:,43:73]+delta],dim=-1);mask=(action==0).all(-1)
  assert torch.equal(result[mask],value[mask])
  return result
 raw.forward=corrected
 zero=d['action'].reshape(-1,30)[:4]*0;ss=d['state'].reshape(-1,73)[:4];cc=d['context'].reshape(-1,34)[:4]
 znew=raw(ss,zero,cc);zold=original_forward(ss,zero,cc)
 print('ZERO_PROBE',json.dumps({'max_abs':float((znew-zold).abs().max()),'finite':bool(torch.isfinite(znew).all() and torch.isfinite(zold).all()),'shape_new':list(znew.shape),'shape_old':list(zold.shape)}),flush=True)
 zero_repeat_error=float((znew-zold).abs().max());assert zero_repeat_error<=64*torch.finfo(znew.dtype).eps*max(1.,float(zold.abs().max()))
 updated=model(d['state'].reshape(-1,73),d['action'].reshape(-1,30),d['context'].reshape(-1,34)).reshape(240,4,73)
 error_before=residual;error_after=(loss.target[:,43:73]-loss.derivative(updated)[:,43:73]).double()
 assert coefficient.abs().max()>0
 assert not torch.equal(updated[:,:,43:73],pred[:,:,43:73])
 assert error_after.square().mean()<error_before.square().mean()
 np.savez(a.output/'fit_coefficients.npz',joint_action_residual=coefficient.cpu().numpy())
 receipt=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),fit_conditions=240,parameters=900,rule='Unregularized least squares on fit joint-velocity response residual only; additive physical joint velocity per physical normalized action; no bias. Root velocity response not fitted.',fit_mse_before=float(error_before.square().mean()),fit_mse_after=float(error_after.square().mean()),zero_action_same_forward_exact=True,zero_repeat_max_abs=zero_repeat_error,policy_acceptance=False)
 (a.output/'fit.json').write_text(json.dumps(receipt,indent=2))
 for collection in a.collection:
  mapping=json.loads((collection/'control_mapping.json').read_text());assert mapping['state_joint_names']==cfg['extra']['tw_state_joint_names'] and mapping['action_joint_names']==cfg['extra']['tw_action_joint_names']
  trainer=SimpleNamespace(world_model=model,device='cuda:0',env=SimpleNamespace(step_dt=.02),spec=SimpleNamespace(joint_names=mapping['state_joint_names']));rows=[]
  for row in json.loads((collection/'results.json').read_text())['rows']:
   with np.load(collection/row['trace']) as z:metrics=audit_world_model(trainer,{k:z[k] for k in z.files})
   rows.append(dict(row,world_model=metrics))
  (a.output/(collection.name+'_after.json')).write_text(json.dumps(dict(checkpoint_sha256=receipt['checkpoint_sha256'],rows=rows,diagnostic_only=True,scope='Frozen additive joint action residual, fit coefficients never use these evaluation labels; not a deployable artifact or policy.'),indent=2));print('AUDITED',collection.name,flush=True)
 assert all(torch.equal(v,before[k]) for k,v in raw.state_dict().items());receipt['base_weights_unchanged']=True
 (a.output/'integrity.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
