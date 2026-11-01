"""Fixed fit-only actual-objective gradients plus excluded diagnostic probes."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,torch
from supermimic_geometry_model import load_geometry_artifact
from supermimic_yaw_canonical import apply_coordinate_contract
from supermimic_motion_windows import load_motion_fit_windows
from supermimic_motion_root_response import MotionRootResponseLoss
from supermimic_antithetic_response import AntitheticResponseLoss
from supermimic_root_state_response import RootStateResponseLoss
from supermimic_root_pose_response import RootPoseResponseLoss
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir();r=a.root;pair=r/'motion_root_mix_fit_v1'
forbidden=set()
for directory,filename in [(r/'motion_pd_collection_cache_v2/full','trace'),(r/'motion_root_velocity_branches_v1/serial_full','file')]:
 for row in json.loads((directory/'results.json').read_text())['rows']:
  if row['split']=='validation':forbidden.add((directory/row[filename]).resolve())
original_load=np.load;opened=[]
def monitored_load(path,*args,**kwargs):
 resolved=Path(path).resolve();assert resolved not in forbidden
 opened.append(str(resolved));return original_load(path,*args,**kwargs)
np.load=monitored_load
fit=json.loads((pair/'control/fit/fit_input.json').read_text());keys=['state','action','context','next_state'];pools={};sources=[]
for group,directory,rows in [('trajectory','static_perturb_v1',fit['trajectory_files']),('one','orthogonal_branches_v1',fit['one_step_files'])]:
 pieces=[]
 for row in rows:
  path=(r/directory/row['file']).resolve();assert path.is_relative_to((r/directory).resolve());assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in keys}
  if group=='trajectory':
   assert np.array_equal(d['next_state'][:-1],d['state'][1:])
   for j in range(len(d['state'])-7):pieces.append({k:(v[j] if k=='state' else v[j:j+8]) for k,v in d.items()})
  else:pieces.extend([{k:v[j] for k,v in d.items()} for j in range(len(d['state']))])
  sources.append(dict(file=str(path),sha256=row['sha256']))
 pools[group]={k:torch.tensor(np.stack([x[k] for x in pieces]),device='cuda:0') for k in keys}
assert len(pools['trajectory']['state'])==3249 and len(pools['one']['state'])==1200
reports=[]
for arm in ['control','candidate']:
 path=pair/arm/'fit/world_model_calibrated.pt';c=torch.load(path,map_location='cpu',weights_only=False);m,predict=load_geometry_artifact(c);rm=apply_coordinate_contract(m,c);m.eval()
 for v in m.parameters():v.requires_grad_(False)
 for v in m.net.parameters():v.requires_grad_(True)
 params=list(m.net.named_parameters());mapping=json.loads((r/'motion_pd_collection_cache_v2/full/control_mapping.json').read_text())
 motion,meta=load_motion_fit_windows(r/'motion_pd_collection_cache_v2/full',r/'motion_pd_collection_cache_v2/full_audit.json',mapping)
 h1,h1meta=load_motion_fit_windows(r/'motion_pd_collection_cache_v2/full',r/'motion_pd_collection_cache_v2/full_audit.json',mapping,window=1);h1={k:(v if k=='state' else v[:,0]) for k,v in h1.items()}
 action=AntitheticResponseLoss(r/'orthogonal_branches_v1',r/'antithetic_response_fit_v1/pair_integrity.json');static=RootStateResponseLoss(r/'root_velocity_branches_v1',c['cfg']);dynamic=MotionRootResponseLoss(r/'motion_root_velocity_branches_v1/serial_full',c['cfg']);pose=RootPoseResponseLoss(r/'root_pose_branches_v1',c['cfg'])
 def value(pred,target):
  e=m.state_norm.normalize(pred)-m.state_norm.normalize(target)
  return sum(e[:,sl].square().mean() for sl in [slice(7,10),slice(10,13),slice(43,73)])/3
 for batch in range(4):
  offset=100*batch
  def generator(seed):return torch.Generator(device='cuda:0').manual_seed(seed+offset)
  tg,og,mg,hg=[generator(seed) for seed in [42,43,48,51]]
  action.generator.manual_seed(44+offset);static.generator.manual_seed(45+offset);pose.generator.manual_seed(46+offset);dynamic.generator.manual_seed(52+offset)
  ti=torch.randint(3249,(64,),device='cuda:0',generator=tg);mi=torch.randint(716,(32,),device='cuda:0',generator=mg)
  traj={k:torch.cat([v[ti][:32],motion[k][mi]]) for k,v in pools['trajectory'].items()}
  pred=traj['state'];rollout=0;velocity=0
  for j in range(8):
   pred=predict(pred,traj['action'][:,j],traj['context'][:,j]);rollout+=m.state_loss(pred,traj['next_state'][:,j])/8;velocity+=value(pred,traj['next_state'][:,j])/8
  oi=torch.randint(1200,(64,),device='cuda:0',generator=og);hi=torch.randint(828,(32,),device='cuda:0',generator=hg)
  one={k:torch.cat([v[oi][:32],h1[k][hi]]) for k,v in pools['one'].items()}
  out=predict(one['state'],one['action'],one['context']);ri=dynamic.sample()
  static_loss=static(rm,used_batch=8 if arm=='candidate' else None);motion_loss=dynamic(rm,ri)
  losses=dict(base=.5*(rollout+m.state_loss(out,one['next_state'])),velocity=.5*(velocity+value(out,one['next_state'])),action_response=action(rm)[0],root_response=.5*(static_loss+motion_loss) if arm=='candidate' else static_loss,pose_response=pose(rm))
  primary=list(losses)
  losses.update(static_response_probe=static_loss,motion_response_probe=motion_loss,static_h1_probe=m.state_loss(out[:32],one['next_state'][:32]),motion_h1_probe=m.state_loss(out[32:],one['next_state'][32:]))
  gradients={};terms={}
  for name,loss in losses.items():
   gs=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True);pieces=[]
   for (key,_),g in zip(params,gs):
    g=g.detach().clone()
    if key=='0.weight':g[:,:2]=0
    assert torch.isfinite(g).all();pieces.append(g.flatten())
   flat=torch.cat(pieces);gradients[name]=flat;terms[name]=dict(loss=float(loss),gradient_norm=float(flat.norm()))
  total=sum(gradients[k] for k in primary);pairs=[]
  for x,y in itertools.combinations(gradients,2):
   den=float(gradients[x].norm()*gradients[y].norm());pairs.append(dict(left=x,right=y,cosine=float(torch.dot(gradients[x],gradients[y]))/den if den else None))
  report=dict(arm=arm,batch=batch,seed_offset=offset,primary_labels=primary,terms=terms,pairs=pairs,total_gradient_norm=float(total.norm()),clip_factor=min(1.,c['cfg']['wm_grad_clip']/max(float(total.norm()),1e-30)))
  reports.append(report);print(json.dumps(report),flush=True)
  del gradients,losses,total,gs,pieces,flat,pred,out,rollout,velocity,static_loss,motion_loss
 for k,v in m.state_dict().items():assert torch.equal(v.cpu(),c['world_model'][k]),k
 sources.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 del m,predict,rm,c
np.load=original_load
out=dict(reports=reports,sources=sources,opened_npz_files=sorted(set(opened)),validation_arrays_loaded=False,weights_unchanged=True,optimizer_updates=0,scope='Four fixed fit batches per final arm. Only primary_labels sum to actual objective; probes excluded. Static response uses actual16/8 conditions for control/candidate. Motion responses use8. Local gradient evidence only, not universal conflict or physical acceptance.')
(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
