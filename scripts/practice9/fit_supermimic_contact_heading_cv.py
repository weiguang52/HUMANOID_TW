"""Source-family held-out H1 heading-residual diagnostic; never a policy artifact."""
import argparse,json,hashlib,copy,datetime
from pathlib import Path
import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from supermimic_geometry_model import load_geometry_artifact
from supermimic_precision_contract import validate
from supermimic_yaw_canonical import canonicalize_state,rotate_vector
from supermimic_contact_heading import contact_heading_features,CONTRACT
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queries',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--response-only',action='store_true');p.add_argument('--fold-plan',type=Path);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
plan=json.loads((a.queries/'query_plan.json').read_text());audit=json.loads((a.collection/'independent_audit.json').read_text());assert audit['passed'] and not audit['smoke'] and audit.get('diagnostic_cone','pyramidal')=='pyramidal'
f=a.collection/'transitions.npz';assert hashlib.sha256(f.read_bytes()).hexdigest()==audit['transitions_sha256']
with np.load(f,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
expanded=plan.get('contract')=='expanded_fit_native_world_yaw_symmetry_v1'
if expanded:
 assert a.response_only and a.fold_plan
 covariance=json.loads((a.collection/'yaw_covariance_audit.json').read_text())
 assert all(x['failed_1e_5']==0 for x in covariance['summary'] if x['angle'] in [90,180]), 'Quarter-turn physics gate failed; do not fit fourfold features'
selected=np.flatnonzero(d['repeat']==0);assert len(selected)==(2880 if expanded else 480)
families=np.array([str(plan['rows'][int(d['query'][i])]['source_family']) for i in selected]);unique=sorted(set(families));assert len(unique)==(24 if expanded else 4) and all(x['split']=='fit' for x in plan['rows'])
if a.fold_plan:
 folds=json.loads(a.fold_plan.read_text());assert folds['contract']=='source_family_heading_fourfold_v1'
 assert hashlib.sha256(Path(folds['manifest']).read_bytes()).hexdigest()==folds['manifest_sha256']
 manifest=json.loads(Path(plan['manifest']).read_text());assert manifest['parent_source_plan_sha256']==folds['manifest_sha256']
 assert sorted(sum([f['held_out_families'] for f in folds['folds']],[]))==unique
 for f in folds['folds']:assert sorted(set(unique)-set(f['held_out_families']))==f['train_families'] and len(f['held_out_families'])==6
 splits=[(f['name'],f['held_out_families']) for f in folds['folds']]
else:splits=[(family,[family]) for family in unique]
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();artifact=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert validate(artifact)
raw,predict=load_geometry_artifact(artifact);raw.eval();predict.eval()
for param in predict.parameters():param.requires_grad_(False)
with torch.no_grad():
 s=torch.tensor(d['state'][selected],device='cuda:0');action=torch.tensor(d['action'][selected],device='cuda:0');context=torch.tensor(d['context'][selected],device='cuda:0');n=torch.tensor(d['next_state'][selected],device='cuda:0')
 prediction=torch.cat([predict(s[i:i+64],action[i:i+64],context[i:i+64]) for i in range(0,len(s),64)])
 local,frame=canonicalize_state(s.double());features=contact_heading_features(s.double()).float()
 def velocity(x):return torch.cat([rotate_vector(x[:,7:10].double(),frame['c'],-frame['s']),rotate_vector(x[:,10:13].double(),frame['c'],-frame['s']),x[:,43:73].double()],dim=1).float()
 base=velocity(prediction);target=velocity(n)
 x=torch.cat([local.float(),action,context],dim=1)
 indices=list(range(7,13))+list(range(43,73));scale=(raw.state_norm.var[indices]+raw.state_norm.eps).sqrt().float();scale=torch.where(raw.state_norm.frozen_mask[indices],torch.ones_like(scale),scale)
 assert torch.isfinite(scale).all() and (scale>0).all()
 zero_lookup={(int(q),bool(w)):i for i,(q,w) in enumerate(zip(d['query'][selected],d['warm_source'][selected]))}
 zero_ids=[]
 for i in selected:
  row=plan['rows'][int(d['query'][i])];zero=next(x['query'] for x in plan['rows'] if x['anchor_query']==row['anchor_query'] and x['yaw_degrees']==0)
  zero_ids.append(zero_lookup[zero,bool(d['warm_source'][i])])
 zero_ids=torch.tensor(zero_ids,device='cuda:0')
 assert np.array_equal(families,families[zero_ids.cpu().numpy()])
 if a.response_only:base=base-base[zero_ids];target=target-target[zero_ids]
 y=(target-base)/scale
class Residual(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(139,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,36));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,z,h):return self.net(torch.cat([z,h],dim=1))
def metrics(pred,truth):
 error=pred-truth
 return dict(scaled_mse=float(((error/scale)**2).mean()),root_linear_rmse=float(error[:,:3].square().mean().sqrt()),root_angular_rmse=float(error[:,3:6].square().mean().sqrt()),joint_velocity_rmse=float(error[:,6:].square().mean().sqrt()))
results=[]
for fold,(family,held_families) in enumerate(splits):
 mask=np.isin(families,held_families);train=torch.tensor(np.flatnonzero(~mask),device='cuda:0');test=torch.tensor(np.flatnonzero(mask),device='cuda:0');assert len(train)==3*len(test) and len(test)==len(x)//4
 mean=x[train].mean(0);std=x[train].std(0,unbiased=False).clamp_min(.1);z=(x-mean)/std
 torch.manual_seed(42);template=Residual().cuda();initial=copy.deepcopy(template.state_dict());generator=torch.Generator(device='cpu').manual_seed(42);batches=torch.randint(len(train),(1000,128),generator=generator);batch_sha=hashlib.sha256(batches.numpy().tobytes()).hexdigest()
 fold_result=dict(held_out_family=family,held_out_families=held_families,train_families=[f for f in unique if f not in held_families],train_records=len(train),test_records=len(test),batch_sha256=batch_sha,baseline=metrics(base[test],target[test]),arms={})
 for arm,enabled in [('constant',False),('heading',True)]:
  model=Residual().cuda();model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=5e-4);h=features if enabled else torch.zeros_like(features)
  with torch.no_grad():assert torch.count_nonzero(model(z,h))==0
  def correction(ids):
   result=model(z[ids],h[ids])
   return result-model(z[zero_ids[ids]],h[zero_ids[ids]]) if a.response_only else result
  losses=[]
  for step in range(1000):
   ids=train[batches[step].to('cuda:0')];loss=(correction(ids)-y[ids]).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward()
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
   if step%100==0:losses.append(float(loss.detach()))
  with torch.no_grad():out=base+correction(torch.arange(len(z),device='cuda:0'))*scale
  report=dict(train=metrics(out[train],target[train]),held_out=metrics(out[test],target[test]),angles={})
  for angle in plan['angles_degrees']:
   ids=torch.tensor([int(i) for i in test.cpu() if plan['rows'][int(d['query'][selected[int(i)]])]['yaw_degrees']==angle],device='cuda:0');report['angles'][str(angle)]=metrics(out[ids],target[ids])
  path=a.output/f'fold{fold}_{arm}.pt';torch.save(dict(kind='offline_contact_heading_h1_residual_diagnostic_not_policy',response_only=a.response_only,contract=CONTRACT,heading_enabled=enabled,state_dict=model.cpu().state_dict(),input_mean=mean.cpu(),input_std=std.cpu(),output_scale=scale.cpu(),optimizer=opt.state_dict(),updates=1000,lr=5e-4,source_checkpoint=str(a.checkpoint),source_sha256=sha,held_out_family=family,held_out_families=held_families,training_families=fold_result['train_families'],batch_sha256=batch_sha,offline_only=True,resumable_policy_training=False,promotion_approved=False),path)
  report['artifact']=str(path);report['loss_every100']=losses;fold_result['arms'][arm]=report
 results.append(fold_result);(a.output/'partial.json').write_text(json.dumps(results,indent=2));print(json.dumps(fold_result),flush=True)
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
for k,v in raw.state_dict().items():assert torch.equal(v.cpu(),artifact['world_model'][k])
summary=dict(fold_plan_sha256=hashlib.sha256(a.fold_plan.read_bytes()).hexdigest() if a.fold_plan else None,response_only=a.response_only,metric_semantics='paired_velocity_response' if a.response_only else 'absolute_velocity',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha,data_sha256=audit['transitions_sha256'],folds=results,all_four_heading_improve=all(f['arms']['heading']['held_out']['scaled_mse']<f['arms']['constant']['held_out']['scaled_mse'] for f in results),optimizer_updates_per_arm_fold=1000,validation_arrays_loaded=False,source_family_isolation=True,original_h8_h32_checked=False,policy_acceptance=False,scope='Development fit-family cross-validation only; all final checkpoints reported, no checkpoint selection. A positive H1 result is insufficient to merge into dynamics or accept control.')
(a.output/'result.json').write_text(json.dumps(summary,indent=2));(a.output/'status').write_text('complete_diagnostic_not_policy');print('CV_COMPLETE',summary['all_four_heading_improve'],flush=True)
