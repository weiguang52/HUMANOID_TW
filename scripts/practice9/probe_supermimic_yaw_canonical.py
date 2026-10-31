"""Fixed read-only coordinate-contract probe; no fitting or policy acceptance."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_geometry_model import configure_offline_geometry
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
from supermimic_yaw_transform import rotate_world_state,state_metrics
from supermimic_yaw_canonical import canonicalize_state,restore_state,YawCanonicalModel,CONTRACT
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;pieces={k:[] for k in ['state','action','context','next_state']};queries=[];sources=[]
for group,relative,count in [('static','static_perturb_v1',16),('motion','motion_pd_collection_cache_v2/full',32)]:
 collection=r/relative;rows=json.loads((collection/'results.json').read_text())['rows'];assert len(rows)==count
 for row in rows:
  path=collection/row['trace'];assert path.resolve().is_relative_to(collection.resolve());digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if 'sha256' in row:assert digest==row['sha256']
  with np.load(path,allow_pickle=False) as z:
   idx=sorted({0,len(z['state'])//2,len(z['state'])-1})
   for k in pieces:pieces[k].append(z[k][idx].copy())
  queries.extend(dict(group=group,split=row['split'],trace=row['trace'],index=i) for i in idx)
  sources.append(dict(group=group,file=row['trace'],sha256=digest))
data={k:torch.tensor(np.concatenate(v),device='cuda:0') for k,v in pieces.items()};assert len(queries)==144
local,frame=canonicalize_state(data['state']);roundtrip=state_metrics(restore_state(local,frame),data['state']);assert roundtrip['max_state_component_error']<=1e-5
np.testing.assert_array_equal(local[:,13:].cpu().numpy(),data['state'][:,13:].cpu().numpy())
raw=data['state'].cpu().double().numpy();oracle=raw.copy();q=raw[:,3:7]/np.linalg.norm(raw[:,3:7],axis=1,keepdims=True);w,x,y,z=q.T
angle=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));c=np.cos(angle);s=np.sin(angle);ch=np.cos(angle/2);sh=np.sin(angle/2)
qq=np.stack([ch*w+sh*z,ch*x+sh*y,ch*y-sh*x,ch*z-sh*w],axis=1);qq=np.where(qq[:,:1]<0,-qq,qq);oracle[:,:2]=0;oracle[:,3:7]=qq
for st in [7,10]:oracle[:,st]=c*raw[:,st]+s*raw[:,st+1];oracle[:,st+1]=-s*raw[:,st]+c*raw[:,st+1]
np.testing.assert_allclose(local.cpu().numpy(),oracle.astype(np.float32),atol=2e-7,rtol=0)
reports=[];saved={k:v.cpu().numpy() for k,v in data.items()};all_passed=True
for label,relative in [('source','standing_wm_calibration_v1/fit/world_model_calibrated.pt'),('height','dynamic_h1_mixture_fit_v1/control/fit/world_model_calibrated.pt'),('motion_mix','motion_h8_mixture_fit_v1/candidate/fit/world_model_calibrated.pt')]:
 path=r/relative;artifact=torch.load(path,map_location='cpu',weights_only=False);cfg=artifact['cfg'];w=artifact['world_model']
 base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
 base=configure_offline_geometry(base,artifact);base.load_state_dict(w);base.eval()
 for param in base.parameters():param.requires_grad_(False)
 original=apply_offline_prediction_contract(base,artifact);control=apply_offline_prediction_contract(YawCanonicalModel(base,False),artifact);candidate=apply_offline_prediction_contract(YawCanonicalModel(base,True),artifact)
 with torch.no_grad():
  for _ in range(3):original(data['state'],data['action'],data['context']);control(data['state'],data['action'],data['context'])
  before=original(data['state'],data['action'],data['context']);disabled=control(data['state'],data['action'],data['context']);pred=candidate(data['state'],data['action'],data['context'])
  checks=[dict(name='disabled_path',metrics=state_metrics(disabled,before))]
  for turns in [0,1,2]:
   value=candidate(rotate_world_state(data['state'],turns),data['action'],data['context']);assert torch.isfinite(value).all();checks.append(dict(name='quarter_turn_'+str(turns),metrics=state_metrics(value,rotate_world_state(pred,turns))))
  antipode=data['state'].clone();antipode[:,3:7]*=-1;value=candidate(antipode,data['action'],data['context']);checks.append(dict(name='quaternion_antipode',metrics=state_metrics(value,pred)))
  for check in checks:check['passed']=check['metrics']['max_state_component_error']<=1e-5
  groups=[]
  for group in ['static','motion']:
   for split in ['fit','validation']:
    mask=torch.tensor([x['group']==group and x['split']==split for x in queries],device='cuda:0')
    groups.append(dict(group=group,split=split,states=int(mask.sum()),initial_function_change=state_metrics(pred[mask],before[mask]),original_archive_error=state_metrics(before[mask],data['next_state'][mask]),canonical_archive_error=state_metrics(pred[mask],data['next_state'][mask])))
  saved[label+'_original']=before.cpu().numpy();saved[label+'_canonical']=pred.cpu().numpy()
 state=data['state'].detach().clone().requires_grad_(True);action=data['action'].detach().clone().requires_grad_(True)
 value=candidate(state,action,data['context']);grads=torch.autograd.grad(value.square().mean(),[state,action]);assert all(torch.isfinite(g).all() for g in grads)
 for k,v in base.state_dict().items():assert torch.equal(v.cpu(),w[k]),k
 passed=all(x['passed'] for x in checks);all_passed=all_passed and passed
 row=dict(model=label,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),structural_checks_passed=passed,checks=checks,groups=groups,finite_state_and_action_gradients=True,weights_unchanged=True)
 reports.append(row);print(json.dumps(dict(model=label,passed=passed,checks=checks)),flush=True)
np.savez_compressed(a.output/'predictions.npz',**saved)
(a.output/'audit.json').write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),contract=CONTRACT,all_structural_checks_passed=all_passed,roundtrip=roundtrip,minimum_heading_projection_squared=float(frame['heading_projection_squared'].min()),numpy_oracle_passed=True,sources=sources,queries=queries,reports=reports,scope='Fixed144 states only. Initial prediction function intentionally changes with unchanged weights. Finite gradients are not a full derivative or control proof; no fitting, policy loading, original gate changes or promotion.'),indent=2))
