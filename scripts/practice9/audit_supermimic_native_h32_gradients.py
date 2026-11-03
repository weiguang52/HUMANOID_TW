"""Fit-only local gradient and hypothetical Adam direction audit, no optimizer steps."""
import argparse,json,hashlib,math,datetime
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_motion_windows import load_motion_fit_windows
from supermimic_native_replay_data import load_native_replay_windows
from supermimic_gradient_projection import project_extra_gradient
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
r=a.root;static_root=r/'static_perturb_v1';mapping=json.loads((static_root/'control_mapping.json').read_text());static_arrays={k:[] for k in ['state','action','context','next_state']};files=[]
for row in json.loads((static_root/'results.json').read_text())['rows']:
 if row['split']!='fit':continue
 f=static_root/row['trace'];files.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 with np.load(f,allow_pickle=False) as z:
  assert np.array_equal(z['next_state'][:-1],z['state'][1:])
  for start in range(len(z['state'])-7):
   static_arrays['state'].append(z['state'][start].copy())
   for key in ['action','context','next_state']:static_arrays[key].append(z[key][start:start+8].copy())
assert len(files)==8 and len(static_arrays['state'])==3249
static={k:torch.tensor(np.stack(v),device='cuda:0') for k,v in static_arrays.items()}
first=torch.load(a.checkpoint[0],map_location='cpu',weights_only=False);cfg=first['cfg']
source=r/'grounded_dynamic_cop_clarabel_init_v1';motion,motion_meta=load_motion_fit_windows(source/'dense_native_collection_v1',source/'dense_native_collection_v1/independent_pd_audit.json',mapping,source_plan_path=source/'diagnostic_references_v1/collection_plan.json')
native,native_meta=load_native_replay_windows(r/'expanded_contact_sources_v1',cfg)
assert len(motion['state'])==783 and len(native['state'])==1588
ids=[torch.randint(n,(16,batch),generator=torch.Generator().manual_seed(seed)) for n,batch,seed in [(3249,32,57),(783,4,58),(1588,1,59)]]
plan=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),static_files=files,motion_input=motion_meta,native_input=native_meta,seeds=[57,58,59],batches=16,indices=[x.tolist() for x in ids],weights=dict(static=.25,replacement=.03125),optimizer_steps=0,validation_arrays_loaded=False,scope='Replacement gradient=.03125*(nativeH32-old4motionH8); protected=.25*(staticH8 state+velocity). Hypothetical directions omit all other actual training losses; first-order only, not causal or acceptance proof.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];models=[]
def dot(x,y):return sum((a.double()*b.double()).sum() for a,b in zip(x,y))
for model_index,path in enumerate(a.checkpoint):
 sha=hashlib.sha256(path.read_bytes()).hexdigest();artifact=torch.load(path,map_location='cpu',weights_only=False);assert 'precision_contract' not in artifact
 model,predict=load_geometry_artifact(artifact);model.eval();params=list(model.net.parameters());assert len(params)==8
 assert artifact['cfg']==cfg
 fit_input=json.loads((path.parent/'fit_input.json').read_text());assert {(x['file'],x['sha256']) for x in fit_input['trajectory_files']}=={(Path(x['file']).name,x['sha256']) for x in files}
 group=artifact['optimizer']['param_groups'][0];assert len(artifact['optimizer']['param_groups'])==1 and len(group['params'])==8 and group['weight_decay']==0 and not group['amsgrad'] and not group.get('maximize',False)
 moments=[artifact['optimizer']['state'][key] for key in group['params']];assert all(int(x['step'])==1000 for x in moments)
 def velocity_loss(pred,target):
  error=model.state_norm.normalize(pred)-model.state_norm.normalize(target)
  return sum(error[:,sl].square().mean() for sl in [slice(7,10),slice(10,13),slice(43,73)])/3
 def rollout(pool,index,horizon):
  index=index.to('cuda:0');state=pool['state'][index];loss=0
  for j in range(horizon):
   state=predict(state,pool['action'][index,j],pool['context'][index,j]);target=pool['next_state'][index,j]
   loss=loss+(model.state_loss(state,target)+velocity_loss(state,target))/horizon
  return loss
 def adam_slope(gradient,protected):
  norm=dot(gradient,gradient).sqrt();clip=min(1.,float(cfg['wm_grad_clip'])/(float(norm)+1e-6));b1,b2=group['betas'];value=0
  for g,gs,state in zip(gradient,protected,moments):
   grad=g.double()*clip;m=state['exp_avg'].to(g.device).double()*b1+(1-b1)*grad;v=state['exp_avg_sq'].to(g.device).double()*b2+(1-b2)*grad.square();step=int(state['step'])+1
   update=-group['lr']*(m/(1-b1**step))/((v/(1-b2**step)).sqrt()+group['eps']);value=value+(gs.double()*update).sum()
  return float(value)
 for batch in range(16):
  static_loss=.25*rollout(static,ids[0][batch],8);gs=torch.autograd.grad(static_loss,params)
  new_loss=rollout(native,ids[2][batch],32);old_loss=rollout(motion,ids[1][batch],8);replacement=.03125*(new_loss-old_loss);gn=torch.autograd.grad(replacement,params)
  assert all(torch.isfinite(x).all() for x in [*gs,*gn]);inner=dot(gs,gn);norms=(dot(gs,gs)*dot(gn,gn)).sqrt();projected,proof=project_extra_gradient(list(gn),list(gs));fixed_preconditioned_slope=0
  for g,protect,state in zip(projected,gs,moments):
   denom=(state['exp_avg_sq'].to(g.device).double()/(1-group['betas'][1]**int(state['step']))).sqrt()+group['eps'];fixed_preconditioned_slope-=group['lr']*(protect.double()*g.double()/denom).sum()
  plain=[x+y for x,y in zip(gs,gn)];projected_total=[x+y for x,y in zip(gs,projected)]
  row=dict(model=model_index,batch=batch,static_loss=float(static_loss),native_h32_loss=float(new_loss),removed_motion_h8_loss=float(old_loss),replacement_loss=float(replacement),gradient_cosine=float(inner/norms) if norms else None,euclidean_projection=proof,fixed_adam_preconditioned_projected_extra_static_slope=float(fixed_preconditioned_slope),hypothetical_adam_static_only_slope=adam_slope(gs,gs),hypothetical_adam_static_plus_replacement_slope=adam_slope(plain,gs),hypothetical_adam_static_plus_euclidean_projected_slope=adam_slope(projected_total,gs))
  rows.append(row);(a.output/'partial.json').write_text(json.dumps(rows,indent=2))
  if (batch+1)%4==0:print(json.dumps(dict(model=model_index,batches=batch+1)),flush=True)
 for k,v in model.state_dict().items():assert torch.equal(v.cpu(),artifact['world_model'][k])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;models.append(dict(file=str(path),sha256=sha));del model,predict,artifact
summary=[]
for model in range(len(models)):
 rs=[x for x in rows if x['model']==model];summary.append(dict(model=model,batches=len(rs),negative_gradient_cosines=sum(x['gradient_cosine']<0 for x in rs),euclidean_projected=sum(x['euclidean_projection']['projected'] for x in rs),projected_extra_adam_scaled_static_increase=sum(x['fixed_adam_preconditioned_projected_extra_static_slope']>0 for x in rs),hypothetical_static_only_increase=sum(x['hypothetical_adam_static_only_slope']>0 for x in rs),hypothetical_projected_total_static_increase=sum(x['hypothetical_adam_static_plus_euclidean_projected_slope']>0 for x in rs)))
(a.output/'audit.json').write_text(json.dumps(dict(models=models,rows=rows,summary=summary,source_weights_unchanged=True,optimizer_steps=0,policy_acceptance=False,scope=plan['scope']),indent=2));print(json.dumps(summary),flush=True)
