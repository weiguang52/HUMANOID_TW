"""Replay the actual complete fit objective at frozen final weights; no training."""
import argparse,ast,copy,hashlib,json,shlex,sys,datetime
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--batches',type=int,default=16)
p.add_argument('--common-descent',action='store_true')
p.add_argument('--component-losses',action='store_true')
p.add_argument('--response-common-descent',action='store_true')
p.add_argument('--adam-metric-common',action='store_true')
p.add_argument('--all-component-common',action='store_true')
a=p.parse_args();assert a.batches==16 and not a.output.exists();a.output.mkdir(parents=True)
source=Path('scripts/practice9/calibrate_supermimic_branch_pair.py')
runner=Path('scripts/practice9/run_supermimic_static_step_guard_pair.sh')
command=next(x for x in runner.read_text().splitlines() if x.startswith('python -u '))
argv=shlex.split(command.split(' > ',1)[0])[3:]
for key,value in [('--updates','4'),('--output',str(a.output/'setup'))]:
 argv[argv.index(key)+1]=value
artifact=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
rows=[];env={};source_state=copy.deepcopy(artifact['world_model'])
def initialize():
 m=env['m'];m.load_state_dict(source_state);env['opt'].load_state_dict(copy.deepcopy(artifact['optimizer']))
 assert len(list(m.net.parameters()))==8
 for key,value in m.state_dict().items():assert torch.equal(value.cpu(),source_state[key])
def dot(xs,ys):return float(sum((x.double()*y.double()).sum() for x,y in zip(xs,ys)))
def capture():
 global rng_before
 generators=[v for v in env.values() if isinstance(v,torch.Generator)]
 generators += [env[k].generator for k in ['response_loss','root_state_loss','root_pose_loss','motion_root_loss'] if env.get(k) is not None]
 assert len({id(g) for g in generators})==len(generators)
 rng_before=[(g,g.get_state().clone()) for g in generators]
def evaluate_full():
 global last_components
 saved=[(g,g.get_state().clone()) for g,_ in rng_before]
 sandbox=env.copy()
 for k,v in sandbox.items():
  if isinstance(v,list) and k.endswith('history'):sandbox[k]=list(v)
 sandbox['point_tally']=dict(env['point_tally'])
 try:
  for g,state in rng_before:g.set_state(state)
  with torch.no_grad():exec(replay_program,sandbox)
  last_components={key:float(sandbox[key]) for key in ['rollout','h1','response_value','root_state_value','root_pose_value','velocity_anchor_value','motion_linear_value'] if sandbox.get(key) is not None}
  return float(sandbox['loss'])
 finally:
  for g,state in saved:g.set_state(state)
def audit():
 m=env['m'];params=list(m.net.parameters());loss=env['loss'];evaluate=env['evaluate_guard_static']
 static=evaluate();gs=torch.autograd.grad(static,params);gf=torch.autograd.grad(loss,params,retain_graph=a.response_common_descent)
 assert all(torch.isfinite(x).all() for x in [*gs,*gf])
 before=[x.detach().clone() for x in params];group=artifact['optimizer']['param_groups'][0]
 assert len(artifact['optimizer']['param_groups'])==1 and group['weight_decay']==0 and not group['amsgrad']
 result=dict(batch=env['update'],static_loss=float(static),full_loss=float(loss),gradient_cosine=dot(gs,gf)/(dot(gs,gs)*dot(gf,gf))**.5,directions={})
 directions=[('full_saved_momentum',gf,False),('full_zero_first_moment',gf,True),('static_saved_momentum',gs,False),('static_zero_first_moment',gs,True)]
 if a.common_descent:directions.append(('common_unit_gradients',gf,False))
 if a.response_common_descent:
  assert a.component_losses and a.common_descent
  gradients=[gs,gf]+[torch.autograd.grad(env[key],params,retain_graph=True) for key in ['response_value','root_state_value','root_pose_value']]
  units=[[v/dot(g,g)**.5 for v in g] for g in gradients]
  gram=torch.tensor([[dot(x,y) for y in units] for x in units],dtype=torch.float64)
  weights=torch.full((5,),.2,dtype=torch.float64)
  for _ in range(256):
   vertex=torch.zeros_like(weights);vertex[torch.argmin(gram@weights)]=1
   difference=vertex-weights;denominator=float(difference@gram@difference)
   if denominator<=1e-30:break
   step=max(0.,min(1.,-float(difference@gram@weights)/denominator))
   weights+=step*difference
  response_direction=[-sum(float(w)*unit[i] for w,unit in zip(weights,units)) for i in range(len(params))]
  result['response_common_weights']=weights.tolist()
  result['response_common_normalized_slopes']=(-gram@weights).tolist()
  directions.append(('response_common_min_norm',gf,False))
  if a.adam_metric_common:
   # P is the inverse Adam RMS. Transform by sqrt(P) so the mapped
   # direction has the same common-descent certificate in parameter space.
   beta2=group['betas'][1];states=artifact['optimizer']['state']
   roots=[]
   for parameter_id,parameter in zip(group['params'],params):
    state=states[parameter_id];step=float(state['step'])
    rms=(state['exp_avg_sq'].to(parameter.device)/(1-beta2**step)).sqrt()+group['eps']
    roots.append(rms.rsqrt())
   transformed=[[v*root for v,root in zip(g,roots)] for g in gradients]
   metric_units=[[v/dot(g,g)**.5 for v in g] for g in transformed]
   metric_gram=torch.tensor([[dot(x,y) for y in metric_units] for x in metric_units],dtype=torch.float64)
   metric_weights=torch.full((5,),.2,dtype=torch.float64)
   for _ in range(256):
    vertex=torch.zeros_like(metric_weights);vertex[torch.argmin(metric_gram@metric_weights)]=1
    difference=vertex-metric_weights;denominator=float(difference@metric_gram@difference)
    if denominator<=1e-30:break
    alpha=max(0.,min(1.,-float(difference@metric_gram@metric_weights)/denominator))
    metric_weights+=alpha*difference
   metric_direction=[-root*sum(float(w)*unit[i] for w,unit in zip(metric_weights,metric_units)) for i,root in enumerate(roots)]
   assert all(torch.isfinite(v).all() for v in metric_direction)
   result['adam_metric_weights']=metric_weights.tolist()
   result['adam_metric_slopes']=[dot(g,metric_direction) for g in gradients]
   directions.append(('adam_metric_common_min_norm',gf,False))

 if a.all_component_common:
  assert a.response_common_descent and a.component_losses
  extra_keys=['rollout','h1','velocity_anchor_value','motion_linear_value']
  all_gradients=gradients+[torch.autograd.grad(env[k],params,retain_graph=True) for k in extra_keys]
  all_norms=[dot(g,g)**.5 for g in all_gradients]
  assert all(n>0 for n in all_norms)
  all_units=[[v/n for v in g] for g,n in zip(all_gradients,all_norms)]
  all_gram=torch.tensor([[dot(x,y) for y in all_units] for x in all_units],dtype=torch.float64)
  all_weights=torch.full((9,),1/9,dtype=torch.float64)
  for _ in range(256):
   vertex=torch.zeros_like(all_weights);vertex[torch.argmin(all_gram@all_weights)]=1
   difference=vertex-all_weights;denominator=float(difference@all_gram@difference)
   if denominator<=1e-30:break
   alpha=max(0.,min(1.,-float(difference@all_gram@all_weights)/denominator))
   all_weights+=alpha*difference
  all_direction=[-sum(float(w)*unit[i] for w,unit in zip(all_weights,all_units)) for i in range(len(params))]
  result['all_component_names']=['static','full','response_value','root_state_value','root_pose_value']+extra_keys
  result['all_component_weights']=all_weights.tolist()
  result['all_component_normalized_slopes']=(-all_gram@all_weights).tolist()
  directions.append(('all_component_common_min_norm',gf,False))
 full_baseline=evaluate_full() if a.common_descent else None
 if a.component_losses:
  assert a.common_descent
  result['component_baseline']=dict(last_components)
 if full_baseline is not None:assert abs(full_baseline-float(loss))<=1e-6*max(1.,abs(float(loss)))
 for name,gradient,reset in directions:
  shadow=[torch.nn.Parameter(x.clone()) for x in before]
  opt=torch.optim.Adam(shadow,lr=group['lr']);opt.load_state_dict(copy.deepcopy(artifact['optimizer']))
  if reset:
   for state in opt.state.values():state['exp_avg'].zero_()
  for parameter,g in zip(shadow,gradient):parameter.grad=g.detach().clone()
  norm=torch.nn.utils.clip_grad_norm_(shadow,env['cfg']['wm_grad_clip']);opt.step()
  delta=[x.detach()-b for x,b in zip(shadow,before)]
  if name in ['common_unit_gradients','response_common_min_norm','adam_metric_common_min_norm','all_component_common_min_norm']:
   direction=all_direction if name=='all_component_common_min_norm' else metric_direction if name=='adam_metric_common_min_norm' else response_direction if name=='response_common_min_norm' else [-(s/dot(gs,gs)**.5+f/dot(gf,gf)**.5) for s,f in zip(gs,gf)]
   scale=(dot(delta,delta)/dot(direction,direction))**.5
   delta=[v*scale for v in direction]
   with torch.no_grad():
    for parameter,start,d in zip(shadow,before,delta):parameter.copy_(start+d)
   delta=[x.detach()-b for x,b in zip(shadow,before)]
  probes=[]
  with torch.no_grad():
   for factor in ([1.,.5,.25,.125,.0625,.03125,.015625,.0078125,0.] if name in ['response_common_min_norm','adam_metric_common_min_norm','all_component_common_min_norm'] else [1.,.5,.25,0.]):
    for parameter,start,end,d in zip(params,before,shadow,delta):parameter.copy_(start if factor==0 else end if factor==1 else start+factor*d)
    value=float(evaluate());assert torch.isfinite(torch.tensor(value))
    full_value=evaluate_full() if a.common_descent else None
    components=dict(last_components) if a.component_losses else None
    probes.append(dict(component_losses=components,factor=factor,loss=value,nonincrease=value<=float(static),full_loss=full_value,both_nonincrease=value<=float(static) and full_value<=full_baseline if a.common_descent else None))
   for parameter,start in zip(params,before):parameter.copy_(start)
  result['directions'][name]=dict(static_slope=dot(gs,delta),full_slope=dot(gf,delta),gradient_norm=float(norm),probes=probes)
 rows.append(result)
 (a.output/'partial.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps(dict(batch=env['update'],directions={k:dict(static_slope=v['static_slope'],full_slope=v['full_slope'],first_acceptable=next(x['factor'] for x in v['probes'] if x['nonincrease'])) for k,v in result['directions'].items()})),flush=True)
# Reuse the exact full objective AST. Stop each batch before backward/real Adam,
# retain all upstream losses and samplers; never execute artifact export.
tree=ast.parse(source.read_text());trial=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(z,ast.For) for z in n.body))
loop=trial.body[0];assert isinstance(loop,ast.For) and ast.unparse(loop.target)=='update'
loop.iter=ast.parse(f'range(1,{a.batches+1})',mode='eval').body
replay_cut=next(i for i,n in enumerate(loop.body) if i>0 and isinstance(n,ast.If) and ast.unparse(n.test)=='point_tally is not None')
replay_program=compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(loop.body[:replay_cut]),type_ignores=[])),str(source),'exec')
cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='loss.backward()')
loop.body=ast.parse('_capture()').body+loop.body[:cut]+ast.parse('_audit()').body
prefix=tree.body[:tree.body.index(trial)]
program=ast.Module(body=prefix+ast.parse('_initialize()').body+[loop],type_ignores=[]);ast.fix_missing_locations(program)
plan=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(a.checkpoint),checkpoint_sha256=sha,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),batches=a.batches,common_descent=a.common_descent,component_losses=a.component_losses,response_common_descent=a.response_common_descent,adam_metric_common=a.adam_metric_common,all_component_common=a.all_component_common,argv=argv,scope='All actual fit losses/samplers at frozen final weights; shadow Adam only, no source optimizer steps. Reset-first-moment retains second moments and step count; hypothetical diagnosis, not proposed accepted optimizer. Finite probes measure same fit-static batch only. Existing setup may load reference metadata; no validation loss used. Not a historical reconstruction of the 745 zero steps or closed-loop proof.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
sys.argv=[str(source)]+argv;sys.path.insert(0,str(source.parent.resolve()))
env.update(__name__='__main__',__file__=str(source),_initialize=initialize,_audit=audit,_capture=capture)
exec(compile(program,str(source),'exec'),env)
for key,value in env['m'].state_dict().items():assert torch.equal(value.cpu(),source_state[key])
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
summary={name:dict(static_ascent=sum(r['directions'][name]['static_slope']>0 for r in rows),full_ascent=sum(r['directions'][name]['full_slope']>0 for r in rows),identity_only=sum(not any(p['nonincrease'] for p in r['directions'][name]['probes'][:-1]) for r in rows),both_nonzero=sum(any(p['both_nonincrease'] for p in r['directions'][name]['probes'][:-1]) for r in rows) if a.common_descent else None) for name in rows[0]['directions']}
(a.output/'audit.json').write_text(json.dumps(dict(plan=plan,rows=rows,summary=summary,source_weights_unchanged=True,source_optimizer_steps=0,policy_acceptance=False),indent=2))
print(json.dumps(summary),flush=True)
