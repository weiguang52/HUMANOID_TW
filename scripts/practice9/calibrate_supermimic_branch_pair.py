"""Matched-budget final-only offline comparison; sole variable is H1 data source."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__)
for key in ['parent','collection','branches','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--sole-geometry',type=int,choices=[0,1]);p.add_argument('--geometry-asset',type=Path);p.add_argument('--geometry-contract-probe',type=Path)
p.add_argument('--root-pose-coefficient',type=float);p.add_argument('--root-pose-scale-audit',type=Path);p.add_argument('--root-pose-collection',type=Path)
p.add_argument('--quaternion-conditioning',type=int,choices=[0,1])
p.add_argument('--xy-invariance',type=int,choices=[0,1])
p.add_argument('--velocity-anchor-coefficient',type=float,choices=[0.,1.])
p.add_argument('--root-state-coefficient',type=float,choices=[0.,1.]);p.add_argument('--root-state-collection',type=Path)
p.add_argument('--response-scale-audit',type=Path)
p.add_argument('--response-coefficient',type=float);p.add_argument('--response-fit-audit',type=Path);p.add_argument('--updates',type=int,default=1000)
p.add_argument('--arm',choices=['control','candidate'],required=True);p.add_argument('--pose-integration-fit-audit',type=Path);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
assert a.updates in [4,1000]
if a.sole_geometry is not None:
 assert a.root_pose_coefficient is not None and a.quaternion_conditioning is None and a.geometry_asset is not None and a.geometry_contract_probe is not None
else:assert a.geometry_asset is None and a.geometry_contract_probe is None
if a.quaternion_conditioning is not None:assert a.xy_invariance==1
if a.root_pose_coefficient is not None:
 assert math.isfinite(a.root_pose_coefficient) and a.root_pose_coefficient>=0
 assert a.root_pose_collection is not None and a.xy_invariance==1 and a.quaternion_conditioning is None
 if a.root_pose_coefficient not in [0.,1.]:assert a.root_pose_scale_audit is not None and a.sole_geometry==1
else:assert a.root_pose_scale_audit is None and a.root_pose_collection is None
if a.xy_invariance is not None:assert a.velocity_anchor_coefficient==1 and a.root_state_coefficient==1 and a.response_coefficient==1
if a.velocity_anchor_coefficient is not None:assert a.root_state_coefficient==1 and a.response_coefficient==1 and a.pose_integration_fit_audit is not None
if a.root_state_coefficient is not None:assert a.root_state_collection is not None and a.response_coefficient==1 and a.pose_integration_fit_audit is not None
else:assert a.root_state_collection is None
if a.response_coefficient is not None:assert math.isfinite(a.response_coefficient) and a.response_coefficient>=0
if a.response_coefficient not in [None,0.,1.]:assert a.response_scale_audit is not None
if a.response_coefficient is not None:assert a.arm=='candidate' and a.response_fit_audit is not None
else:assert a.updates==1000 and a.response_fit_audit is None
torch.manual_seed(42);c=torch.load(a.parent,map_location='cpu',weights_only=False);assert c['kind']=='offline_dynamics_calibration_not_policy_checkpoint' and c['calibration_plan']['window']==8 and c['calibration_updates']==1000
w=c['world_model'];cfg=c['cfg'];plan=dict(seed=42,updates=1000,batch=128,window=8,arm=a.arm,trajectory_batch=64,one_step_batch=64,targets_per_update=576,lr=.0001,source_checkpoint_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),one_step_source='original_fit' if a.arm=='control' else 'orthogonal_fit_including_zero',frozen=['termination','normalizers'],policy_artifact=False,selection='Final1000 only; no checkpoint selection, no validation fit')
plan['updates']=a.updates
if a.response_coefficient is not None:plan.update(response_coefficient=a.response_coefficient,response_batch=16,response_fit_audit_sha256=hashlib.sha256(a.response_fit_audit.read_bytes()).hexdigest(),differing_variable='Response coefficient0 vs1; identical four-endpoint sampling and forward budget')
scale_measurement=None
if a.response_scale_audit is not None:
 scale_measurement=json.loads(a.response_scale_audit.read_text())
 raw_path=Path(scale_measurement['source_measurement_file'])
 assert raw_path.resolve().is_relative_to(a.response_scale_audit.parent.resolve())
 assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==scale_measurement['source_measurement_sha256']
 raw=json.loads(raw_path.read_text())
 assert raw['coefficient']==scale_measurement['coefficient']==a.response_coefficient
 assert raw['parent_sha256']==plan['source_checkpoint_sha256']
 assert scale_measurement['integration_audit_sha256']==hashlib.sha256(a.pose_integration_fit_audit.read_bytes()).hexdigest()
 assert scale_measurement['pair_audit_sha256']==hashlib.sha256(a.response_fit_audit.read_bytes()).hexdigest()
 assert abs(raw['coefficient']*raw['response_loss']-raw['base_loss'])<1e-15
 plan.update(response_scale_audit_sha256=hashlib.sha256(a.response_scale_audit.read_bytes()).hexdigest(),response_weight_rule='Frozen-parent fit-only base/response ratio; no validation selection')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
def load(path):
 with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in ['state','action','context','next_state']}
 n=len(d['state'])
 for k,width in [('state',73),('action',30),('context',34),('next_state',73)]:assert d[k].shape==(n,width) and np.isfinite(d[k]).all()
 return d
mapping=json.loads((a.collection/'control_mapping.json').read_text());bm=json.loads((a.branches/'control_mapping.json').read_text())
for key,extra in [('state_joint_names','tw_state_joint_names'),('action_joint_names','tw_action_joint_names')]:assert mapping[key]==bm[key]==cfg['extra'][extra]
for key in ['offset','scale']:np.testing.assert_allclose(np.array(bm[key]),np.broadcast_to(np.array(mapping[key]),np.array(bm[key]).shape),rtol=0,atol=1e-7)
traj={k:[] for k in ['state','action','context','next_state']};original=[];used=[];files=[]
rows=json.loads((a.collection/'results.json').read_text())['rows']
for row in rows:
 if row['split']!='fit':continue
 path=(a.collection/row['trace']).resolve();assert path.is_relative_to(a.collection.resolve());d=load(path);n=len(d['state']);assert np.array_equal(d['next_state'][:-1],d['state'][1:]);original.append(d);used.append(row['trace']);files.append(dict(file=row['trace'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 for start in range(n-8+1):
  traj['state'].append(d['state'][start])
  for k in ['action','context','next_state']:traj[k].append(d[k][start:start+8])
assert len(used)==8
if a.arm=='control':one=original;one_files=files
else:
 integrity=json.loads((a.branches/'integrity.json').read_text());assert integrity['passed'];hashes={x['file']:x['sha256'] for x in integrity['files']}
 report=json.loads((a.branches/'results.json').read_text());assert report['contract']=='branch_transitions_not_episodes_v1';one=[];one_files=[]
 for row in report['rows']:
  if row['split']!='fit':continue
  path=(a.branches/row['file']).resolve();assert path.is_relative_to(a.branches.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==hashes[row['file']]
  one.append(load(path));one_files.append(dict(file=row['file'],sha256=digest))
 assert len(one)==20
if scale_measurement is not None:
 expected={('trajectory',v['file']):v['sha256'] for v in files}
 expected.update({('branch',v['file']):v['sha256'] for v in one_files})
 assert expected=={(v['source'],v['file']):v['sha256'] for v in scale_measurement['sources']}
traj={k:torch.tensor(np.stack(v),device='cuda:0') for k,v in traj.items()};one={k:torch.tensor(np.concatenate([d[k] for d in one]),device='cuda:0') for k in traj}
assert len(one['state'])==(3305 if a.arm=='control' else 1200)
(a.output/'fit_input.json').write_text(json.dumps(dict(traces=used,windows=len(traj['state']),trajectory_files=files,one_step_count=len(one['state']),one_step_files=one_files),indent=2))
m=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();m.load_state_dict(w)
for v in m.parameters():v.requires_grad_(False)
for v in m.net.parameters():v.requires_grad_(True)
predictor=m;prediction_contract=None
if a.pose_integration_fit_audit is not None:
 assert a.arm=='candidate'
 from supermimic_pose_integration_diagnostic import PoseVelocityModel,CONTRACT
 audit=json.loads(a.pose_integration_fit_audit.read_text());assert audit['fit_frames']==4505
 recorded={(r['source'],r['file']):r['sha256'] for r in audit['sources'] if r['split']=='fit'}
 expected={('trajectory',r['file']):r['sha256'] for r in files};expected.update({('branch',r['file']):r['sha256'] for r in one_files});assert recorded==expected
 prediction_contract=dict(contract=CONTRACT,alpha=audit['alpha'],dt=audit['plan']['dt'],fit_audit_sha256=hashlib.sha256(a.pose_integration_fit_audit.read_bytes()).hexdigest(),fit_frames=4505)
 predictor=PoseVelocityModel(m,prediction_contract);plan['prediction_contract']=prediction_contract
 plan['differing_variable']='Pose reconstructed from predicted velocities during both H8 and H1 fitting; same parent/data/samplers/budget as previous candidate control'
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
if scale_measurement is not None:
 plan['differing_variable']='Fit-only measured response coefficient vs1; same explicit prediction structure and data budget'
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
response_loss=None
if a.response_coefficient is not None:
 from supermimic_antithetic_response import AntitheticResponseLoss
 response_loss=AntitheticResponseLoss(a.branches,a.response_fit_audit)
plan['base_state_targets_per_update']=576
plan['prediction_endpoints_per_update']=576+(64 if a.response_coefficient is not None else 0)+(64 if a.root_state_coefficient is not None else 0)
root_state_loss=None
if a.root_state_coefficient is not None:
 from supermimic_root_state_response import RootStateResponseLoss
 root_state_loss=RootStateResponseLoss(a.root_state_collection,cfg)
 plan.update(root_state_coefficient=a.root_state_coefficient,root_state_batch=16,root_state_fit_files=root_state_loss.files,root_state_collection_plan_sha256=hashlib.sha256((a.root_state_collection/'plan.json').read_bytes()).hexdigest(),differing_variable='Root-state Jacobian coefficient0 vs1; equal extra64 endpoints per update')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
root_pose_loss=None
if a.root_pose_coefficient is not None:
 from supermimic_root_pose_response import RootPoseResponseLoss
 root_pose_loss=RootPoseResponseLoss(a.root_pose_collection,cfg)
 plan.update(root_pose_coefficient=a.root_pose_coefficient,root_pose_batch=16,
  root_pose_fit_files=root_pose_loss.files,root_pose_collection_plan_sha256=hashlib.sha256((a.root_pose_collection/'plan.json').read_bytes()).hexdigest(),
  root_pose_rule='Central response times declared physical base perturbation divided by fixed parent output std; no validation fitting')
 plan['prediction_endpoints_per_update']+=64
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))

def velocity_value_loss(pred,target):
 error=m.state_norm.normalize(pred)-m.state_norm.normalize(target)
 return (error[:,7:10].square().mean()+error[:,10:13].square().mean()+error[:,43:73].square().mean())/3
if a.velocity_anchor_coefficient is not None:
 plan.update(velocity_anchor_coefficient=a.velocity_anchor_coefficient,velocity_anchor_rule='Equal parent-normalized root linear/root angular/joint velocity group MSE, half H8/half H1; same predictions and targets',differing_variable='Velocity value anchor coefficient0 vs1 only; no extra forward endpoints')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
if a.xy_invariance is not None:
 sample_ids=torch.linspace(0,len(traj['state'])-1,64,device='cuda:0').long()
 query=traj['state'][sample_ids].clone();query[:,:2]=0
 query_action=traj['action'][sample_ids,0];query_context=traj['context'][sample_ids,0]
 with torch.no_grad():
  for _ in range(5):m(query,query_action,query_context)
  before=m(query,query_action,query_context)
  if a.xy_invariance:
   first=m.net[0];reference=m.state_norm.normalize(query)[0,:2]
   first.bias.add_(first.weight[:,:2]@reference);first.weight[:,:2].zero_()
  after=m(query,query_action,query_context)
  initial_error=float((after-before).abs().max());assert initial_error<=1e-5
 if a.xy_invariance:
  def mask_xy_gradient(gradient):
   result=gradient.clone();result[:,:2]=0;return result
  m.net[0].weight.register_hook(mask_xy_gradient)
 plan.update(xy_invariance=bool(a.xy_invariance),xy_reference_physical=[0.,0.],xy_reference_initial_max_error=initial_error,differing_variable='Dynamics-only horizontal translation equivariance0 vs1; standard serialized weights, no inference hook')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
geometry_contract=None;geometry_frozen={}
if a.sole_geometry is not None:
 from supermimic_geometry_model import install_geometry_input,GEOMETRY_CONTRACT
 probe=json.loads(a.geometry_contract_probe.read_text());assert probe['passed']
 assert probe['source_sha256']==plan['source_checkpoint_sha256']
 geometry_contract=dict(contract=GEOMETRY_CONTRACT,enabled=bool(a.sole_geometry),height_scale_m=.001,
  ground_height_m=0,asset=str(a.geometry_asset.resolve()),asset_sha256=hashlib.sha256(a.geometry_asset.read_bytes()).hexdigest())
 with torch.no_grad():
  for _ in range(5):m(query,query_action,query_context)
  geometry_before=m(query,query_action,query_context)
  install_geometry_input(m,geometry_contract,cfg)
  if a.xy_invariance:m.net[0].weight.register_hook(mask_xy_gradient)
  for _ in range(5):m(query,query_action,query_context)
  geometry_error=float((m(query,query_action,query_context)-geometry_before).abs().max())
  assert geometry_error<=1e-5,geometry_error
 geometry_frozen={k:v.detach().clone() for k,v in m.state_dict().items() if k.startswith('net.0.') and k not in ['net.0.weight','net.0.bias']}
 plan.update(geometry_contract=geometry_contract,geometry_initial_error=geometry_error,
  geometry_probe_sha256=hashlib.sha256(a.geometry_contract_probe.read_bytes()).hexdigest())
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
q_hook=None
q_training_state=None
if a.quaternion_conditioning is not None:
 fit_q=torch.cat([torch.tensor(d['state'][:,3:7],device='cuda:0') for d in original]+[one['state'][:,3:7]])
 assert fit_q.shape==(4505,4)
 q_mean=fit_q.mean(0);q_std=fit_q.std(0,unbiased=False)
 assert torch.isfinite(q_std).all() and (q_std>0).all()
 sample_ids=torch.linspace(0,len(traj['state'])-1,64,device='cuda:0').long()
 q_query=traj['state'][sample_ids];q_action=traj['action'][sample_ids,0];q_context=traj['context'][sample_ids,0]
 def q_forward_hook(module,args):
  x=args[0].clone();x[...,3:7]=(x[...,3:7]-q_mean)/q_std
  return (x,)
 with torch.no_grad():
  for _ in range(5):m(q_query,q_action,q_context)
  q_before=m(q_query,q_action,q_context)
  if a.quaternion_conditioning:
   first=m.net[0];raw=first.weight[:,3:7].clone()
   first.bias.add_(raw@q_mean);first.weight[:,3:7].mul_(q_std)
   q_hook=first.register_forward_pre_hook(q_forward_hook)
  for _ in range(5):m(q_query,q_action,q_context)
  q_error=float((m(q_query,q_action,q_context)-q_before).abs().max())
  assert q_error<=1e-5,q_error
 plan.update(quaternion_conditioning=bool(a.quaternion_conditioning),quaternion_fit_frames=4505,
  quaternion_mean=q_mean.tolist(),quaternion_std=q_std.tolist(),quaternion_initial_max_error=q_error,
  differing_variable='Training-only quaternion feature coordinates0 vs1; raw-coordinate inference export')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
if a.root_pose_coefficient is not None:
 plan['differing_variable']='Root-pose response coefficient0 vs1 only; identical fit conditions and extra64 endpoints'
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
if geometry_contract is not None:
 plan['differing_variable']='Explicit geometry input flag and loss coefficients recorded; paired variable defined in experiment plan'
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
if a.root_pose_scale_audit is not None:
 scale=json.loads(a.root_pose_scale_audit.read_text())
 assert scale['weights_unchanged'] and scale['new_optimizer_updates']==0
 assert scale['primary_h8_windows']==3249 and scale['primary_h1_states']==1200 and scale['pose_conditions']==48
 assert scale['source_checkpoint_sha256']==plan['source_checkpoint_sha256']
 assert scale['coefficient']==a.root_pose_coefficient
 assert scale['primary_gradient_norm']>0 and scale['pose_gradient_norm']>0
 assert abs(scale['coefficient']-scale['primary_gradient_norm']/scale['pose_gradient_norm'])<1e-12
 assert scale['initialization_geometry_contract']==geometry_contract
 assert scale['prediction_contract']==prediction_contract
 expected={('trajectory',v['file']):v['sha256'] for v in files}
 expected.update({('branch',v['file']):v['sha256'] for v in one_files})
 recorded={(v['group'],Path(v['file']).name):v['sha256'] for v in scale['sources']}
 assert expected==recorded
 assert scale['pose_fit_files']==root_pose_loss.files
 plan.update(root_pose_scale_audit_sha256=hashlib.sha256(a.root_pose_scale_audit.read_bytes()).hexdigest(),
  root_pose_scale_rule=scale['rule'],differing_variable='Root pose coefficient1 vs one frozen full-fit parent gradient-norm ratio')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
opt=torch.optim.Adam(m.net.parameters(),lr=plan['lr']);tg=torch.Generator(device='cuda:0').manual_seed(42);og=torch.Generator(device='cuda:0').manual_seed(43)
update=0
try:
 for update in range(1,a.updates+1):
  idx=torch.randint(len(traj['state']),(64,),device='cuda:0',generator=tg);pred=traj['state'][idx];rollout=0
  velocity_rollout=0
  for j in range(8):
   pred=predictor(pred,traj['action'][idx,j],traj['context'][idx,j]);rollout=rollout+m.state_loss(pred,traj['next_state'][idx,j])/8
   if a.velocity_anchor_coefficient is not None:velocity_rollout=velocity_rollout+velocity_value_loss(pred,traj['next_state'][idx,j])/8
  oi=torch.randint(len(one['state']),(64,),device='cuda:0',generator=og);one_pred=predictor(one['state'][oi],one['action'][oi],one['context'][oi]);h1=m.state_loss(one_pred,one['next_state'][oi]);loss=.5*rollout+.5*h1
  velocity_anchor_value=None
  if a.velocity_anchor_coefficient is not None:
   velocity_anchor_value=.5*(velocity_rollout+velocity_value_loss(one_pred,one['next_state'][oi]))
   loss=loss+a.velocity_anchor_coefficient*velocity_anchor_value
  response_value=None
  if response_loss is not None:
   response_value,root_response,joint_response=response_loss(m)
   loss=loss+a.response_coefficient*response_value
  root_state_value=None
  if root_state_loss is not None:
   root_state_value=root_state_loss(m)
   loss=loss+a.root_state_coefficient*root_state_value
  root_pose_value=None
  if root_pose_loss is not None:
   root_pose_value=root_pose_loss(m)
   loss=loss+a.root_pose_coefficient*root_pose_value
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.net.parameters(),cfg['wm_grad_clip']);assert torch.isfinite(gn);opt.step()
  if update%100==0 or update==a.updates:print(json.dumps(dict(update=update,loss=float(loss),rollout_loss=float(rollout),one_step_loss=float(h1),gradient_norm=float(gn),response_loss=float(response_value) if response_value is not None else None,root_state_loss=float(root_state_value) if root_state_value is not None else None,root_pose_loss=float(root_pose_value) if root_pose_value is not None else None,velocity_anchor_loss=float(velocity_anchor_value) if velocity_anchor_value is not None else None)),flush=True)
 if q_hook is not None:
  with torch.no_grad():
   for _ in range(5):m(q_query,q_action,q_context)
   trained_prediction=m(q_query,q_action,q_context)
   q_training_state={k:v.detach().cpu().clone() for k,v in m.net[0].state_dict().items()}
   first=m.net[0];first.weight[:,3:7].div_(q_std)
   first.bias.sub_(first.weight[:,3:7]@q_mean)
   q_hook.remove();q_hook=None
   for _ in range(5):m(q_query,q_action,q_context)
   export_error=float((m(q_query,q_action,q_context)-trained_prediction).abs().max())
   assert export_error<=1e-5,export_error
  plan['quaternion_raw_export_max_error']=export_error
  (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
 if a.xy_invariance:assert torch.count_nonzero(m.net[0].weight[:,:2])==0
 for key,value in geometry_frozen.items():assert torch.equal(m.state_dict()[key],value),key
 for key,value in m.state_dict().items():
  assert torch.isfinite(value).all()
  if not key.startswith('net.'):assert torch.equal(value.cpu(),w[key]),key
 artifact=dict(kind='offline_dynamics_calibration_not_policy_checkpoint',world_model=m.state_dict(),cfg=cfg,calibration_plan=plan,calibration_updates=a.updates,optimizer=opt.state_dict())
 if a.quaternion_conditioning:
  artifact['training_coordinate_optimizer']=artifact.pop('optimizer')
  artifact['training_coordinate_first_linear']=q_training_state
  artifact['optimizer_coordinate_contract']='Conditioned quaternion features; not raw-coordinate resume optimizer'
 if prediction_contract is not None:
  from supermimic_pose_integration_diagnostic import POSE_KIND
  artifact['kind']=POSE_KIND;artifact['prediction_contract']=prediction_contract
 if a.updates!=1000:artifact['kind']='offline_smoke_not_policy_checkpoint'
 if geometry_contract is not None:
  from supermimic_geometry_model import GEOMETRY_KIND
  artifact.update(kind=GEOMETRY_KIND,geometry_contract=geometry_contract,smoke_only=a.updates!=1000)
 torch.save(artifact,a.output/'world_model_calibrated.pt')
 (a.output/'status').write_text('calibrated_pending_audit\n')
except BaseException:
 plan['interrupted_quaternion_hook_active']=q_hook is not None
 plan['interrupted_geometry_contract']=geometry_contract
 torch.save(dict(kind='offline_interrupted_not_policy',world_model=m.state_dict(),calibration_plan=plan,calibration_updates=update,optimizer=opt.state_dict(),prediction_contract=prediction_contract),a.output/'interrupt.pt');(a.output/'status').write_text('failed\n');raise
