"""Frozen actual complete-objective directions; fit probes, no source updates."""
import argparse,ast,copy,datetime,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--root-common-probe',action='store_true')
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
source=Path('scripts/practice9/calibrate_supermimic_static_coupled_pair.py')
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
artifact=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
rows=[];W={};E=None
before=None;optimizer_before=None
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
 return x==y
def dot(x,y):return float(sum((u.double()*v.double()).sum() for u,v in zip(x,y)))
def initialize():
 global before,optimizer_before
 W['initialize']()
 E['m'].load_state_dict(artifact['world_model']);E['opt'].load_state_dict(copy.deepcopy(artifact['optimizer']))
 before={k:v.clone() for k,v in E['m'].state_dict().items()};optimizer_before=copy.deepcopy(E['opt'].state_dict())
def audit():
 params=list(E['m'].net.parameters());additional=W['additional'];model=E['response_model']
 idx=additional.history[-1]['indices']
 def common():
  pred=model(*[additional.data[k][idx].reshape(-1,w) for k,w in [('state',73),('action',30),('context',34)]]).reshape(8,4,73)
  norm=model.state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std)
  error=(pred-additional.data['next_state'][idx])/std
  return {name:.5*error[:,:,sl].mean(1).square().mean() for name,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]}
 values={'full_with_joint':E['loss'],'full_without_joint':E['loss']-additional.last_joint,'joint_response':additional.last_joint}
 values.update({'common_'+k:v for k,v in common().items()})
 if a.root_common_probe:values['full_with_joint_and_root_common']=values['full_with_joint']+values['common_linear']+values['common_angular']
 grads={k:torch.autograd.grad(v,params,retain_graph=True) for k,v in values.items()}
 assert all(torch.isfinite(g).all() for gs in grads.values() for g in gs)
 row={'batch':E['update'],'indices':idx,'losses':{k:float(v) for k,v in values.items()},'probes':{}}
 weights=[v.detach().clone() for v in params]
 for name in ['full_without_joint','full_with_joint']+(['full_with_joint_and_root_common'] if a.root_common_probe else []):
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
out={'timestamp':datetime.datetime.now().astimezone().isoformat(),'checkpoint':str(a.checkpoint),'sha256':sha,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'root_common_probe':a.root_common_probe,'fit_input':W['additional'].meta,'rows':rows,'weights_unchanged':True,'source_optimizer_unchanged':True,'source_optimizer_updates':0,'policy_acceptance':False,'scope':'Frozen full actual objective, original fit samplers, saved Adam/lr/clipping. Paired omission of joint response on shadow copies only. New first16 fit draws at frozen weights, not historical1000-update reconstruction or closed-loop acceptance.'}
(a.output/'audit.json').write_text(json.dumps(out,indent=2))
