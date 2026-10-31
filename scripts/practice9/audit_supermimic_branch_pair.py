"""All held temporal branch states and swapped-world derivatives; final models only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch,mujoco
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic.algo.kinematics import DifferentiableKinematics,build_kinematic_tree
p=argparse.ArgumentParser(description=__doc__)
for key in ['pair-root','branches','asset','parent']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--pose-integration-fit-audit',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();output=a.output or a.pair_root/'branch_validation.json';assert not output.exists()
paths={'parent':a.parent,**{arm:a.pair_root/arm/'fit/world_model_calibrated.pt' for arm in ['control','candidate']}}
models={};configs={}
for name,path in paths.items():
 c=torch.load(path,map_location='cpu',weights_only=False);assert c['kind']=='offline_dynamics_calibration_not_policy_checkpoint' and c['calibration_updates']==1000;cfg=c['cfg'];w=c['world_model'];configs[name]=cfg
 assert cfg['extra']['tw_mjcf_sha256']==hashlib.sha256(a.asset.read_bytes()).hexdigest()
 model=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();model.load_state_dict(w);model.eval();models[name]=model
if a.pose_integration_fit_audit is not None:
 from supermimic_pose_integration_diagnostic import PoseIntegrationDiagnostic
 models['candidate_independent']=models['candidate'];paths['candidate_independent']=paths['candidate']
 models['candidate']=PoseIntegrationDiagnostic(models['candidate'],a.pose_integration_fit_audit,hashlib.sha256(paths['candidate'].read_bytes()).hexdigest())
mapping=json.loads((a.branches/'control_mapping.json').read_text())
for cfg in configs.values():
 assert cfg['extra']['tw_state_joint_names']==mapping['state_joint_names'] and cfg['extra']['tw_action_joint_names']==mapping['action_joint_names']
mm=mujoco.MjModel.from_xml_path(str(a.asset));names=[mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,mm.nbody)];assert names[0]=='base_link'
fk=DifferentiableKinematics(build_kinematic_tree(mm,names,mapping['state_joint_names'])).cuda();feet=[names.index(s+'_foot') for s in ['left','right']]
for side in ['left','right']:
 gid=mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_GEOM,side+'_foot_collision');assert gid>=0;np.testing.assert_allclose(mm.geom_quat[gid],[1,0,0,0],rtol=0,atol=1e-7)
def tensor(x):return torch.as_tensor(x,device='cuda:0',dtype=torch.float32)
@torch.no_grad()
def score(s):
 _,q=fk(s[:,:3],s[:,3:7],s[:,13:43]);q=torch.nn.functional.normalize(q[:,feet],dim=-1);return 2*q[:,:,1:3].square().sum((-1,-2))
@torch.no_grad()
def normal_xy(s):
 _,q=fk(s[:,:3],s[:,3:7],s[:,13:43]);q=torch.nn.functional.normalize(q[:,feet],dim=-1);w,x,y,z=q.unbind(-1)
 return torch.stack([2*(x*z+w*y),2*(y*z-w*x)],dim=-1).reshape(len(s),4)
@torch.no_grad()
def oracle_correct(pred,zero_pred,zero_actual):
 corrected=pred+(zero_actual-zero_pred);corrected=corrected.clone();corrected[:,3:7]=torch.nn.functional.normalize(corrected[:,3:7],dim=-1);return corrected
def metrics(pred,truth):
 q=torch.nn.functional.normalize(pred[:,3:7],dim=-1);t=torch.nn.functional.normalize(truth[:,3:7],dim=-1);t=torch.where((q*t).sum(-1,keepdim=True)<0,-t,t);angle=4*torch.asin((torch.linalg.vector_norm(q-t,dim=-1)/2).clamp(0,1))
 out={'root_orientation_p95_rad':float(torch.quantile(angle,.95))}
 for name,sl in [('root_position',slice(0,3)),('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:out[name+'_rmse']=float((pred[:,sl]-truth[:,sl]).square().mean().sqrt())
 return out
integrity=json.loads((a.branches/'integrity.json').read_text());assert integrity['passed'];hashes={r['file']:r['sha256'] for r in integrity['files']};rows=json.loads((a.branches/'results.json').read_text())['rows'];loaded={};truths=[];predictions={k:[] for k in models};sources=[]
for r in rows:
 if r['split']!='validation':continue
 path=(a.branches/r['file']).resolve();assert path.is_relative_to(a.branches.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==hashes[r['file']]
 with np.load(path) as z:d={k:z[k].copy() for k in z.files}
 s,u,ctx,n=[tensor(d[k]) for k in ['state','action','context','next_state']];assert all(torch.isfinite(x).all() for x in [s,u,ctx,n]);pred={}
 for name,model in models.items():
  with torch.no_grad():pred[name]=model(s,u,ctx)
  assert torch.isfinite(pred[name]).all();predictions[name].append(pred[name])
 loaded[(r['warmup'],r['kind'],r.get('epsilon'),r.get('assignment'))]=(d,pred);truths.append(n);sources.append(dict(file=r['file'],sha256=hashes[r['file']]))
assert len(loaded)==20 and sum(len(x) for x in truths)==1200
state_metrics={name:metrics(torch.cat(v),torch.cat(truths)) for name,v in predictions.items()}
def derivative(v0,v1,eps):return ((v0[1::2]-v0[::2])+(v1[::2]-v1[1::2]))/(4*eps)
def diffmetrics(actual,pred):
 x=actual.astype(np.float64).reshape(-1);y=pred.astype(np.float64).reshape(-1);den=np.linalg.norm(x)*np.linalg.norm(y)
 return dict(actual_norm=float(np.linalg.norm(x)),predicted_norm=float(np.linalg.norm(y)),rmse=float(np.sqrt(np.mean((x-y)**2))),cosine=float(x@y/den) if den>1e-15 else None)
conditions=[];all_actual={k:[] for k in ['root_angular_velocity','joint_velocity','sole_cost','sole_normal_xy','sole_cost_oracle_corrected']};all_pred={name:{k:[] for k in all_actual} for name in models}
plan=json.loads((a.branches/'plan.json').read_text())
for warmup in plan['validation_warmups']:
 zd,zp=loaded[(warmup,'zero',None,None)];ztruth=tensor(zd['next_state']);zscore=score(ztruth).cpu().numpy()
 for eps in plan['epsilons']:
  pairs=[loaded[(warmup,'branch',eps,i)] for i in range(2)];actual={};permodel={name:{} for name in models}
  for group,sl in [('root_angular_velocity',slice(10,13)),('joint_velocity',slice(43,73))]:
   actual[group]=derivative(*[x[0]['next_state'][:,sl]-zd['next_state'][:,sl] for x in pairs],eps)
   for name in models:permodel[name][group]=derivative(*[(x[1][name][:,sl]-zp[name][:,sl]).cpu().numpy() for x in pairs],eps)
  actual['sole_cost']=derivative(*[score(tensor(x[0]['next_state'])).cpu().numpy()-zscore for x in pairs],eps)
  for name in models:permodel[name]['sole_cost']=derivative(*[(score(x[1][name])-score(zp[name])).cpu().numpy() for x in pairs],eps)
  actual['sole_normal_xy']=derivative(*[(normal_xy(tensor(x[0]['next_state']))-normal_xy(ztruth)).cpu().numpy() for x in pairs],eps)
  actual['sole_cost_oracle_corrected']=actual['sole_cost']
  for name in models:
   permodel[name]['sole_normal_xy']=derivative(*[(normal_xy(x[1][name])-normal_xy(zp[name])).cpu().numpy() for x in pairs],eps)
   permodel[name]['sole_cost_oracle_corrected']=derivative(*[(score(oracle_correct(x[1][name],zp[name],ztruth))-score(ztruth)).cpu().numpy() for x in pairs],eps)
  for group in actual:
   assert np.isfinite(actual[group]).all();all_actual[group].append(actual[group])
   for name in models:assert np.isfinite(permodel[name][group]).all();all_pred[name][group].append(permodel[name][group])
  conditions.append(dict(warmup=warmup,epsilon=eps,models={name:{g:diffmetrics(actual[g],values[g]) for g in actual} for name,values in permodel.items()}))
derivative_metrics={name:{g:diffmetrics(np.concatenate(all_actual[g]),np.concatenate(all_pred[name][g])) for g in all_actual} for name in models}
checks=[]
for key,old in state_metrics['control'].items():
 new=state_metrics['candidate'][key];checks.append(dict(kind='held_branch_state',metric=key,before=old,after=new,max_ratio=1.05,passed=new<=old*1.05))
for group in ['root_angular_velocity','joint_velocity','sole_cost']:
 old=derivative_metrics['control'][group]['rmse'];new=derivative_metrics['candidate'][group]['rmse'];checks.append(dict(kind='held_branch_derivative',metric=group,before=old,after=new,max_ratio=.8,passed=new<=old*.8))
small=[dict(warmup=x['warmup'],epsilon=x['epsilon']) for x in conditions if x['models']['candidate']['sole_cost']['actual_norm']<=1e-8]
if small:checks.append(dict(kind='insufficient_physical_signal',conditions=small,passed=False))
report=dict(passed=all(x['passed'] for x in checks),scope='All1200 held temporal branch/zero transitions and8 derivative conditions; not independent-motion or control acceptance. Sole score stable2qxySquared. Oracle bias correction uses measured zero response, diagnostic only, never deployed or counted as prediction/control success; no gate tuning or promotion.',model_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},sources=sources,state_metrics=state_metrics,derivative_metrics=derivative_metrics,conditions=conditions,checks=checks,promoted=False)
if a.pose_integration_fit_audit is not None:report['pose_integration_diagnostic']=models['candidate'].metadata
output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(dict(passed=report['passed'],checks=checks,derivative_metrics=derivative_metrics)))
