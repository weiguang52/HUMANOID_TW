"""Fit-only frozen response/common-bias gradients and isolated shadow Adam probes.
Not the full training objective or a reconstruction of its historical updates.
"""
import argparse,ast,copy,datetime,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
r=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics')
src=Path('scripts/practice9/audit_supermimic_calibration_model.py')
tree=ast.parse(src.read_text())
cut=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.ImportFrom) and n.module=='supermimic_pose_integration_diagnostic')
program=compile(ast.fix_missing_locations(ast.Module(body=tree.body[:cut],type_ignores=[])),str(src),'exec')
from supermimic_yaw_canonical import YawCanonicalModel
from supermimic_static_coupled_response import StaticWideCoupledResponseLoss
paths=[r/'shared_hidden_moment_pair_v1/control/fit/world_model_calibrated.pt']
paths += [r/f'static_coupled_response_pair_v1/{arm}/fit/world_model_calibrated.pt' for arm in ['control','candidate']]
def dot(x,y):return float(sum((u.double()*v.double()).sum() for u,v in zip(x,y)))
report={'timestamp':datetime.datetime.now().astimezone().isoformat(),'models':[],'scope':__doc__,'policy_acceptance':False}
for checkpoint in paths:
 sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 artifact=torch.load(checkpoint,map_location='cpu',weights_only=False)
 env={'__name__':'__main__','__file__':str(src)}
 sys.argv=[str(src),'--collection',str(r/'static_perturb_v1'),'--checkpoint',str(checkpoint),'--output',str(a.output/'unused.json')]
 exec(program,env);m=YawCanonicalModel(env['m'],True)
 before={k:v.clone() for k,v in m.state_dict().items()}
 params=[v for n,v in m.named_parameters() if '.net.' in n or n.startswith('net.')];assert len(params)==8
 loss=StaticWideCoupledResponseLoss(r/'wide_static_root_velocity_v1/serial_full',env['cfg'],1)
 def common(indices):
  pred=m(*[loss.data[k][indices].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(-1,4,73)
  std=(m.state_norm.var+m.state_norm.eps).sqrt();std=torch.where(m.state_norm.frozen_mask,torch.ones_like(std),std)
  error=(pred-loss.data['next_state'][indices])/std
  return {name:.5*error[:,:,sl].mean(1).square().mean() for name,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]}
 rows=[]
 for batch in range(16):
  total=loss(m);indices=loss.history[-1]['indices'];values={'root_response':loss.last_root,'joint_response':loss.last_joint,'combined_response':total}
  values.update({'common_'+k:v for k,v in common(indices).items()})
  grads={k:torch.autograd.grad(v,params,retain_graph=True) for k,v in values.items()}
  assert all(torch.isfinite(g).all() for gs in grads.values() for g in gs)
  row={'batch':batch+1,'indices':indices,'losses':{k:float(v) for k,v in values.items()},'joint_gradient_pairs':{},'isolated_shadow_adam':{}}
  for k,g in grads.items():
   j=grads['joint_response'];row['joint_gradient_pairs'][k]={'dot':dot(j,g),'cosine':dot(j,g)/max((dot(j,j)*dot(g,g))**.5,1e-300)}
  weights=[v.detach().clone() for v in params]
  for name in ['root_response','joint_response','combined_response']:
   shadow=[torch.nn.Parameter(v.clone()) for v in weights]
   opt=torch.optim.Adam(shadow,lr=artifact['optimizer']['param_groups'][0]['lr'])
   opt.load_state_dict(copy.deepcopy(artifact['optimizer']))
   for v,g in zip(shadow,grads[name]):v.grad=g.detach().clone()
   torch.nn.utils.clip_grad_norm_(shadow,env['cfg']['wm_grad_clip']);opt.step()
   delta=[v.detach()-w for v,w in zip(shadow,weights)]
   probe={'slopes':{k:dot(g,delta) for k,g in grads.items()}}
   try:
    with torch.no_grad():
     for v,s in zip(params,shadow):v.copy_(s)
     probe['common_after']={k:float(v) for k,v in common(indices).items()}
   finally:
    with torch.no_grad():
     for v,w in zip(params,weights):v.copy_(w)
   row['isolated_shadow_adam'][name]=probe
  assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
  rows.append(row);print('AUDITED',sha,batch+1,flush=True)
  (a.output/'partial.json').write_text(json.dumps(dict(report,current_rows=rows),indent=2))
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==sha
 report['models'].append({'checkpoint':str(checkpoint),'sha256':sha,'rows':rows,'weights_unchanged':True,'source_optimizer_steps':0,'fit_only':True,'input':loss.meta})
 (a.output/'partial.json').write_text(json.dumps(report,indent=2))
 del m,env,loss,grads,values,artifact,before;torch.cuda.empty_cache()
(a.output/'audit.json').write_text(json.dumps(report,indent=2))
