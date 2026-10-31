"""Read-only model covariance on every predeclared motion query; no fitting."""
import argparse,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_geometry_model import configure_offline_geometry
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
from supermimic_yaw_transform import rotate_world_state,state_metrics
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
r=a.root;collection=r/'motion_pd_collection_cache_v2/full';rows=json.loads((collection/'results.json').read_text())['rows'];assert len(rows)==32
pieces={k:[] for k in ['state','action','context','next_state']};queries=[];sources=[]
for row in rows:
 path=collection/row['trace'];assert path.resolve().is_relative_to(collection.resolve());sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==row['sha256']
 with np.load(path,allow_pickle=False) as z:
  idx=sorted({0,len(z['state'])//2,len(z['state'])-1})
  for k in pieces:pieces[k].append(z[k][idx].copy())
 queries.extend(dict(trace=row['trace'],index=i,split=row['split']) for i in idx)
 sources.append(dict(file=row['trace'],sha256=sha))
data={k:torch.tensor(np.concatenate(v),device='cuda:0') for k,v in pieces.items()};assert len(queries)==96
for turns in [0,1,2]:
 s=data['state'];rot=rotate_world_state(s,turns);oracle=s.cpu().numpy().copy();raw=s.cpu().numpy()
 if turns:
  mat=np.array([[0,-1],[1,0]]) if turns==1 else -np.eye(2)
  for st in [0,7,10]:oracle[:,st:st+2]=raw[:,st:st+2]@mat.T
  c,h=(math.sqrt(.5),math.sqrt(.5)) if turns==1 else (0.,1.)
  w,x,y,z=raw[:,3:7].T;oracle[:,3:7]=np.stack([c*w-h*z,c*x-h*y,c*y+h*x,c*z+h*w],axis=1)
 np.testing.assert_allclose(rot.cpu().numpy(),oracle,atol=2e-7,rtol=0)
 assert state_metrics(rotate_world_state(rot,-turns),s)['max_state_component_error']<=5e-7
reports=[];saved={k:v.cpu().numpy() for k,v in data.items()}
for label,relative in [('source','standing_wm_calibration_v1/fit/world_model_calibrated.pt'),('height','dynamic_h1_mixture_fit_v1/control/fit/world_model_calibrated.pt'),('motion_mix','motion_h8_mixture_fit_v1/candidate/fit/world_model_calibrated.pt')]:
 path=r/relative;digest=hashlib.sha256(path.read_bytes()).hexdigest();artifact=torch.load(path,map_location='cpu',weights_only=False);cfg=artifact['cfg'];w=artifact['world_model']
 model=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda()
 model=configure_offline_geometry(model,artifact);model.load_state_dict(w);model.eval();predict=apply_offline_prediction_contract(model,artifact)
 with torch.no_grad():
  original=predict(data['state'],data['action'],data['context'])
  for turns in [0,1,2]:
   output=predict(rotate_world_state(data['state'],turns),data['action'],data['context']);expected=rotate_world_state(original,turns);target=rotate_world_state(data['next_state'],turns)
   assert torch.isfinite(output).all();saved[label+'_turn'+str(turns)]=output.cpu().numpy()
   for split in ['fit','validation']:
    mask=torch.tensor([x['split']==split for x in queries],device='cuda:0')
    reports.append(dict(model=label,checkpoint_sha256=digest,yaw_degrees=90*turns,split=split,states=int(mask.sum()),covariance_error=state_metrics(output[mask],expected[mask]),transformed_archive_error_not_native=state_metrics(output[mask],target[mask])))
  for k,v in model.state_dict().items():assert torch.equal(v.cpu(),w[k]),k
 print(label,[x for x in reports if x['model']==label and x['split']=='validation'],flush=True)
np.savez_compressed(a.output/'predictions.npz',**saved)
(a.output/'audit.json').write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),sources=sources,queries=queries,reports=reports,weights_unchanged=True,transform_numpy_oracle_passed=True,predictions_sha256=hashlib.sha256((a.output/'predictions.npz').read_bytes()).hexdigest(),scope='Mathematical covariance only. Rotated archive endpoints are not verified native physics; no fitting, coefficient choice, acceptance gate or policy promotion.'),indent=2))
