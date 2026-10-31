"""Fit-only state-loss decomposition; no weight selection or fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from supermimic.algo.networks import WorldModel,WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract
p=argparse.ArgumentParser(description=__doc__)
for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir()
plan=dict(scope='All fit transitions only; decompose exact existing loss, no parameter selection, policy training or causal claim',non_quaternion_dimensions=69,groups={'root_position':3,'root_linear_velocity':3,'root_angular_velocity':3,'joint_position':30,'joint_velocity':30},quaternion_multiplier=4)
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
sources={};provenance={};trace_records={}
for source,folder in [('trajectory','static_perturb_v1'),('branch','orthogonal_branches_v1')]:
 root=a.root/folder;records=[];files=[]
 for row in json.loads((root/'results.json').read_text())['rows']:
  if row['split']!='fit':continue
  path=root/(row['trace'] if source=='trajectory' else row['file'])
  with np.load(path,allow_pickle=False) as z:records.append({k:z[k].copy() for k in ['state','action','context','next_state']})
  files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 trace_records[source]=records
 sources[source]={k:torch.tensor(np.concatenate([d[k] for d in records]),device='cuda:0') for k in records[0]};provenance[source]=files
seq={k:[] for k in ['state','action','context','next_state']}
for d in trace_records['trajectory']:
 assert np.array_equal(d['next_state'][:-1],d['state'][1:])
 for i in range(len(d['state'])-7):
  seq['state'].append(d['state'][i])
  for k in ['action','context','next_state']:seq[k].append(d[k][i:i+8])
seq={k:torch.tensor(np.stack(v),device='cuda:0') for k,v in seq.items()};assert len(seq['state'])==3249
scales={}
models={'independent':a.root/'standing_branch_pair_v1/candidate/fit/world_model_calibrated.pt','integrated':a.root/'pose_velocity_fit_v1/fit/world_model_calibrated.pt'};rows=[]
for name,path in models.items():
 c=torch.load(path,map_location='cpu',weights_only=False);w=c['world_model'];cfg=c['cfg'];base=WorldModel(WorldModelCfg(state_dim=73,action_dim=30,context_dim=34,goal_dim=w['goal_norm.mean'].numel(),hidden_dims=tuple(cfg['wm_hidden_dims']),termination_hidden=tuple(cfg['wm_term_hidden_dims']))).cuda();base.load_state_dict(w);base.eval();model=apply_offline_prediction_contract(base,c)
 scales[name]=(base.state_norm.var+base.state_norm.eps).sqrt().cpu().tolist()
 for source,d in sources.items():
  with torch.no_grad():pred=model(d['state'],d['action'],d['context']);truth=d['next_state'];e=base.state_norm.normalize(pred)-base.state_norm.normalize(truth);groups={}
  for group,sl in [('root_position',slice(0,3)),('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:
   groups[group]=dict(dimensions=sl.stop-sl.start,normalized_mse=float(e[:,sl].square().mean()),loss_contribution=float(e[:,sl].square().sum(-1).mean()/69),physical_rmse=float((pred[:,sl]-truth[:,sl]).square().mean().sqrt()))
  q=pred[:,3:7];t=truth[:,3:7];sign=torch.where((q*t).sum(-1,keepdim=True)>=0,1.,-1.);qloss=float(4*(sign*q-t).square().mean());total=float(base.state_loss(pred,truth));parts=sum(v['loss_contribution'] for v in groups.values())+qloss;assert abs(parts-total)<=max(1e-10,abs(total)*1e-5)
  for v in groups.values():v['fraction_total']=v['loss_contribution']/total if total else None
  rows.append(dict(model=name,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=source,frames=len(pred),groups=groups,quaternion_loss=qloss,quaternion_fraction=qloss/total if total else None,total_loss=total,reconstruction_abs_error=abs(parts-total)))
 accum={k:0. for k in plan['groups']};quat=0.;loss=0.
 with torch.no_grad():
  pred=seq['state']
  for j in range(8):
   pred=model(pred,seq['action'][:,j],seq['context'][:,j]);truth=seq['next_state'][:,j];assert torch.isfinite(pred).all();e=base.state_norm.normalize(pred)-base.state_norm.normalize(truth)
   for group,sl in [('root_position',slice(0,3)),('root_linear_velocity',slice(7,10)),('root_angular_velocity',slice(10,13)),('joint_position',slice(13,43)),('joint_velocity',slice(43,73))]:accum[group]+=float(e[:,sl].square().sum(-1).mean()/69)/8
   q=pred[:,3:7];t=truth[:,3:7];sign=torch.where((q*t).sum(-1,keepdim=True)>=0,1.,-1.);quat+=float(4*(sign*q-t).square().mean())/8;loss+=float(base.state_loss(pred,truth))/8
 assert abs(sum(accum.values())+quat-loss)<=max(1e-10,loss*1e-5)
 groups={k:dict(loss_contribution=v,fraction_total=v/loss) for k,v in accum.items()}
 rows.append(dict(model=name,source='trajectory_H8',windows=3249,groups=groups,quaternion_loss=quat,quaternion_fraction=quat/loss,total_loss=loss))
 branch=next(r for r in rows if r['model']==name and r['source']=='branch');mixed={k:.5*accum[k]+.5*branch['groups'][k]['loss_contribution'] for k in accum};mq=.5*quat+.5*branch['quaternion_loss'];ml=.5*loss+.5*branch['total_loss']
 assert abs(sum(mixed.values())+mq-ml)<=max(1e-10,ml*1e-5)
 rows.append(dict(model=name,source='actual_mixed_objective',groups={k:dict(loss_contribution=v,fraction_total=v/ml) for k,v in mixed.items()},quaternion_loss=mq,quaternion_fraction=mq/ml,total_loss=ml))
report=dict(plan=plan,frozen_state_normalizer_std=scales,sources=provenance,rows=rows,promoted=False);(a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
for r in rows:print(r['model'],r['source'],r['total_loss'],{k:round(v['fraction_total']*100,2) for k,v in r['groups'].items()},'quat%',round(r['quaternion_fraction']*100,2))
