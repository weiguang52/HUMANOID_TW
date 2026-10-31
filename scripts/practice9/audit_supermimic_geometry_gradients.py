"""Fit-only fixed-batch gradient conflicts; no optimizer or validation access."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_antithetic_response import AntitheticResponseLoss
from supermimic_root_state_response import RootStateResponseLoss
from supermimic_root_pose_response import RootPoseResponseLoss
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root
pair=r/'sole_geometry_fit_v1';fit=json.loads((pair/'control/fit/fit_input.json').read_text());sources=[]
keys=['state','action','context','next_state'];trajectories=[];one=[]
for group,dirname,rows in [('trajectory','static_perturb_v1',fit['trajectory_files']),('branch','orthogonal_branches_v1',fit['one_step_files'])]:
 for row in rows:
  path=(r/dirname/row['file']).resolve();assert path.is_relative_to((r/dirname).resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in keys}
  (trajectories if group=='trajectory' else one).append(d)
  sources.append(dict(group=group,file=str(path),sha256=row['sha256']))
traj={k:[] for k in keys}
for d in trajectories:
 assert np.array_equal(d['next_state'][:-1],d['state'][1:])
 for start in range(len(d['state'])-7):
  traj['state'].append(d['state'][start])
  for k in keys[1:]:traj[k].append(d[k][start:start+8])
traj={k:torch.tensor(np.stack(v),device='cuda:0') for k,v in traj.items()}
one={k:torch.tensor(np.concatenate([d[k] for d in one]),device='cuda:0') for k in keys}
assert len(traj['state'])==3249 and len(one['state'])==1200
reports=[]
for arm in ['control','candidate']:
 path=pair/arm/'fit/world_model_calibrated.pt';c=torch.load(path,map_location='cpu',weights_only=False)
 assert c['calibration_updates']==1000 and not c['smoke_only']
 m,predictor=load_geometry_artifact(c);m.eval()
 for v in m.parameters():v.requires_grad_(False)
 for v in m.net.parameters():v.requires_grad_(True)
 params=list(m.net.named_parameters());assert params[0][0]=='0.weight'
 action=AntitheticResponseLoss(r/'orthogonal_branches_v1',r/'antithetic_response_fit_v1/pair_integrity.json')
 velocity=RootStateResponseLoss(r/'root_velocity_branches_v1',c['cfg'])
 pose=RootPoseResponseLoss(r/'root_pose_branches_v1',c['cfg'])
 def value(pred,target):
  e=m.state_norm.normalize(pred)-m.state_norm.normalize(target)
  return sum(e[:,sl].square().mean() for sl in [slice(7,10),slice(10,13),slice(43,73)])/3
 for batch_id in range(4):
  seed=100*batch_id
  tg=torch.Generator(device='cuda:0').manual_seed(42+seed);og=torch.Generator(device='cuda:0').manual_seed(43+seed)
  action.generator.manual_seed(44+seed);velocity.generator.manual_seed(45+seed);pose.generator.manual_seed(46+seed)
  idx=torch.randint(len(traj['state']),(64,),device='cuda:0',generator=tg)
  oi=torch.randint(len(one['state']),(64,),device='cuda:0',generator=og)
  pred=traj['state'][idx];rollout=0;vrollout=0
  for j in range(8):
   pred=predictor(pred,traj['action'][idx,j],traj['context'][idx,j])
   rollout+=m.state_loss(pred,traj['next_state'][idx,j])/8
   vrollout+=value(pred,traj['next_state'][idx,j])/8
  p1=predictor(one['state'][oi],one['action'][oi],one['context'][oi])
  losses=dict(base=.5*(rollout+m.state_loss(p1,one['next_state'][oi])),velocity_value=.5*(vrollout+value(p1,one['next_state'][oi])),action_response=action(m)[0],root_velocity_response=velocity(m),root_pose_response=pose(m))
  gradients={};terms={}
  for label,loss in losses.items():
   values=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True,allow_unused=True)
   pieces=[];feature_norm=0.
   for (name,param),g in zip(params,values):
    g=torch.zeros_like(param) if g is None else g.detach().clone()
    if name=='0.weight':g[:,:2]=0;feature_norm=float(g[:,137:].norm())
    assert torch.isfinite(g).all();pieces.append(g.reshape(-1))
   flat=torch.cat(pieces);gradients[label]=flat
   terms[label]=dict(loss=float(loss),gradient_norm=float(flat.norm()),feature_gradient_norm=feature_norm)
  total=sum(gradients.values());total_norm=float(total.norm())
  pairs=[]
  for x,y in itertools.combinations(gradients,2):
   g,h=gradients[x],gradients[y];den=float(g.norm()*h.norm())
   pairs.append(dict(left=x,right=y,cosine=float(torch.dot(g,h))/den if den else None))
  reports.append(dict(arm=arm,batch_id=batch_id,seed_offset=seed,terms=terms,pairs=pairs,total_gradient_norm=total_norm,clip_factor=min(1.,c['cfg']['wm_grad_clip']/max(total_norm,1e-30))))
  print(json.dumps(dict(arm=arm,batch_id=batch_id,terms=terms,pairs=pairs)),flush=True)
  del gradients,losses,total,values,pieces,flat,pred,p1,rollout,vrollout
 for key,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][key]),key
 del m,predictor,c
out=dict(sources=sources,reports=reports,weights_unchanged=True,scope='Four fixed fit-only batches per final model. Projected zero XY gradient as in training; no optimizer, validation access, coefficient selection or causal proof. Batch-local conflicts are not an acceptance result.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
