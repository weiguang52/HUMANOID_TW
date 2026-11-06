"""Frozen actual complete-objective directions; fit probes, no source updates."""
import argparse,ast,copy,datetime,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--root-common-probe',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
source=Path('scripts/practice9/calibrate_supermimic_static_common_pair.py')
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
artifact=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
rows=[];W={};E=None
before=None;optimizer_before=None
dense=None;dense_rng=None
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
def dot(x,y):return float(sum((u.double()*v.double()).sum() for u,v in zip(x,y)))
def initialize():
 global before,optimizer_before,dense,dense_rng
 W['initialize']()
 E['m'].load_state_dict(artifact['world_model']);E['opt'].load_state_dict(copy.deepcopy(artifact['optimizer']))
 before={k:v.clone() for k,v in E['m'].state_dict().items()};optimizer_before=copy.deepcopy(E['opt'].state_dict())
 import numpy as np
 dense={k:torch.tensor(np.concatenate([d[k] for d in E['original']]),device='cuda:0') for k in E['original'][0]}
 assert len(dense['state'])==3305
 dense_rng=torch.Generator(device='cuda:0').manual_seed(65)
def audit():
 params=list(E['m'].net.parameters());additional=W['additional'];model=E['response_model']
 idx=additional.history[-1]['indices']
 def common():
  pred=model(*[additional.data[k][idx].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  norm=model.state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std)
  error=(pred-additional.data['next_state'][idx])/std
  return {name:.5*error[:,:,sl].mean(1).square().mean() for name,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]}
 ids=torch.randint(3305,(32,),device='cuda:0',generator=dense_rng)
 batch={k:v[ids] for k,v in dense.items()}
 pred=E['predictor'](batch['state'],batch['action'],batch['context'])
 dense_state=E['m'].state_loss(pred,batch['next_state'])
 dense_velocity=E['velocity_value_loss'](pred,batch['next_state'])
 dense_value=.5*dense_state+.5*dense_velocity
 values={'full_base':E['loss'],'full_plus_dense':E['loss']+dense_value,'dense_state':dense_state,'dense_velocity':dense_velocity,'dense_weighted':dense_value}
 values.update({'common_'+k:v for k,v in common().items()})
 grads={k:torch.autograd.grad(v,params,retain_graph=True) for k,v in values.items()}
 assert all(torch.isfinite(g).all() for gs in grads.values() for g in gs)
 row={'batch':E['update'],'indices':idx,'dense_indices':ids.cpu().tolist(),'losses':{k:float(v) for k,v in values.items()},'probes':{}}
 weights=[v.detach().clone() for v in params]
 for name in ['full_base','full_plus_dense']:
  shadow=[torch.nn.Parameter(v.clone()) for v in weights]
  opt=torch.optim.Adam(shadow,lr=artifact['optimizer']['param_groups'][0]['lr']);opt.load_state_dict(copy.deepcopy(artifact['optimizer']))
  for v,g in zip(shadow,grads[name]):v.grad=g.detach().clone()
  gn=torch.nn.utils.clip_grad_norm_(shadow,E['cfg']['wm_grad_clip']);opt.step()
  delta=[v.detach()-w for v,w in zip(shadow,weights)]
  result={'gradient_norm':float(gn),'slopes':{k:dot(g,delta) for k,g in grads.items()}}
  try:
   with torch.no_grad():
    for v,s in zip(params,shadow):v.copy_(s)
    result['common_after']={k:float(v) for k,v in common().items()}
    dp=E['predictor'](batch['state'],batch['action'],batch['context'])
    result['dense_after']=float(.5*E['m'].state_loss(dp,batch['next_state'])+.5*E['velocity_value_loss'](dp,batch['next_state']))
  finally:
   with torch.no_grad():
    for v,w in zip(params,weights):v.copy_(w)
  row['probes'][name]=result
 assert all(torch.equal(v,before[k]) for k,v in E['m'].state_dict().items())
 assert equal(E['opt'].state_dict(),optimizer_before)
 rows.append(row);(a.output/'partial.json').write_text(json.dumps(rows,indent=2));print('AUDITED',E['update'],flush=True)
def execute(tree,filename,env):
 global E
 E=env;E['_initialize']=initialize;E['_audit']=audit
 trial=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(z,ast.For) for z in n.body));loop=trial.body[0]
 cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='loss.backward()')
 loop.iter=ast.parse('range(1,17)',mode='eval').body
 loop.body=loop.body[:cut]+ast.parse('_audit()').body
 program=ast.Module(body=tree.body[:tree.body.index(trial)]+[loop],type_ignores=[])
 exec(compile(ast.fix_missing_locations(program),str(filename),'exec'),E)
tree=ast.parse(source.read_text());last=tree.body[-1];assert isinstance(last,ast.Expr) and ast.unparse(last).startswith('exec(')
last.value=ast.parse('_execute(tree,source,E)',mode='eval').body
sys.argv=[str(source),'--output',str(a.output/'setup'),'--enabled','1','--updates','4']
W.update(__name__='__main__',__file__=str(source),_execute=execute)
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),W)
assert len(rows)==16 and hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
out={'timestamp':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(a.checkpoint),'sha256':sha,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'root_common_probe':a.root_common_probe,'fit_input':W['additional'].meta,'rows':rows,'weights_unchanged':True,'source_optimizer_unchanged':True,'source_optimizer_updates':0,'policy_acceptance':False,'scope':'Frozen full actual objective, original fit samplers, saved Adam/lr/clipping. Paired addition of dense static H1 .5 state+.5 velocity on shadow copies only; 32 fit-only points seed65, 3305 original transitions. Existing root common and joint response retained. New first16 fit draws at frozen weights, not historical1000-update reconstruction or closed-loop acceptance.'}
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
