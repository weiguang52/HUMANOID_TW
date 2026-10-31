"""Matched-budget final-only offline comparison; sole variable is H1 data source."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
p=argparse.ArgumentParser(description=__doc__)
for key in ['parent','collection','branches','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--response-coefficient',type=float,choices=[0.,1.]);p.add_argument('--response-fit-audit',type=Path);p.add_argument('--updates',type=int,default=1000)
p.add_argument('--arm',choices=['control','candidate'],required=True);p.add_argument('--pose-integration-fit-audit',type=Path);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
assert a.updates in [4,1000]
if a.response_coefficient is not None:assert a.arm=='candidate' and a.response_fit_audit is not None and a.pose_integration_fit_audit is None
else:assert a.updates==1000 and a.response_fit_audit is None
torch.manual_seed(42);c=torch.load(a.parent,map_location='cpu',weights_only=False);assert c['kind']=='offline_dynamics_calibration_not_policy_checkpoint' and c['calibration_plan']['window']==8 and c['calibration_updates']==1000
w=c['world_model'];cfg=c['cfg'];plan=dict(seed=42,updates=1000,batch=128,window=8,arm=a.arm,trajectory_batch=64,one_step_batch=64,targets_per_update=576,lr=.0001,source_checkpoint_sha256=hashlib.sha256(a.parent.read_bytes()).hexdigest(),one_step_source='original_fit' if a.arm=='control' else 'orthogonal_fit_including_zero',frozen=['termination','normalizers'],policy_artifact=False,selection='Final1000 only; no checkpoint selection, no validation fit')
plan['updates']=a.updates
if a.response_coefficient is not None:plan.update(response_coefficient=a.response_coefficient,response_batch=16,response_fit_audit_sha256=hashlib.sha256(a.response_fit_audit.read_bytes()).hexdigest(),differing_variable='Response coefficient0 vs1; identical four-endpoint sampling and forward budget')
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
response_loss=None
if a.response_coefficient is not None:
 from supermimic_antithetic_response import AntitheticResponseLoss
 response_loss=AntitheticResponseLoss(a.branches,a.response_fit_audit)
opt=torch.optim.Adam(m.net.parameters(),lr=plan['lr']);tg=torch.Generator(device='cuda:0').manual_seed(42);og=torch.Generator(device='cuda:0').manual_seed(43)
update=0
try:
 for update in range(1,a.updates+1):
  idx=torch.randint(len(traj['state']),(64,),device='cuda:0',generator=tg);pred=traj['state'][idx];rollout=0
  for j in range(8):pred=predictor(pred,traj['action'][idx,j],traj['context'][idx,j]);rollout=rollout+m.state_loss(pred,traj['next_state'][idx,j])/8
  oi=torch.randint(len(one['state']),(64,),device='cuda:0',generator=og);h1=m.state_loss(predictor(one['state'][oi],one['action'][oi],one['context'][oi]),one['next_state'][oi]);loss=.5*rollout+.5*h1
  response_value=None
  if response_loss is not None:
   response_value,root_response,joint_response=response_loss(m)
   loss=loss+a.response_coefficient*response_value
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.net.parameters(),cfg['wm_grad_clip']);assert torch.isfinite(gn);opt.step()
  if update%100==0 or update==a.updates:print(json.dumps(dict(update=update,loss=float(loss),rollout_loss=float(rollout),one_step_loss=float(h1),gradient_norm=float(gn),response_loss=float(response_value) if response_value is not None else None)),flush=True)
 for key,value in m.state_dict().items():
  assert torch.isfinite(value).all()
  if not key.startswith('net.'):assert torch.equal(value.cpu(),w[key]),key
 artifact=dict(kind='offline_dynamics_calibration_not_policy_checkpoint',world_model=m.state_dict(),cfg=cfg,calibration_plan=plan,calibration_updates=a.updates,optimizer=opt.state_dict())
 if a.updates!=1000:artifact['kind']='offline_smoke_not_policy_checkpoint'
 if prediction_contract is not None:
  from supermimic_pose_integration_diagnostic import POSE_KIND
  artifact['kind']=POSE_KIND;artifact['prediction_contract']=prediction_contract
 torch.save(artifact,a.output/'world_model_calibrated.pt')
 (a.output/'status').write_text('calibrated_pending_audit\n')
except BaseException:
 torch.save(dict(kind='offline_interrupted_not_policy',world_model=m.state_dict(),calibration_plan=plan,calibration_updates=update,optimizer=opt.state_dict(),prediction_contract=prediction_contract),a.output/'interrupt.pt');(a.output/'status').write_text('failed\n');raise
