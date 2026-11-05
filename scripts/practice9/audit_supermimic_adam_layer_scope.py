"""Frozen original plus dynamic/dense center objective gradients; no optimizer updates."""
import argparse,ast,copy,hashlib,json,shlex,sys,datetime
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--batches',type=int,default=16)
p.add_argument('--initial-model',action='store_true',help='Audit the exact constructed pre-update model and fresh optimizer from the original runner; checkpoint must be its parent.')
p.add_argument('--gradient-breakdown',action='store_true',help='Record frozen per-layer objective gradient competition; requires baseline-only.')
p.add_argument('--baseline-only',action='store_true',help='Record fixed objective components and sampler hashes without gradient probes.')
p.add_argument('--common-descent',action='store_true')
p.add_argument('--component-losses',action='store_true')
p.add_argument('--response-common-descent',action='store_true')
p.add_argument('--adam-metric-common',action='store_true')
p.add_argument('--all-component-common',action='store_true')
p.add_argument('--project-full-descent',action='store_true')
p.add_argument('--two-batch-static',action='store_true',help='Include independent next-batch static loss: full18 constraints.')
p.add_argument('--two-batch-direction',action='store_true',help='Include next-fit full and seven component gradients; second static is not included.')
p.add_argument('--cross-batch',action='store_true',help='Evaluate every frozen probe on the next fit RNG batch, without selection.')
a=p.parse_args();assert not a.two_batch_static or (a.two_batch_direction and a.cross_batch);assert a.batches==16 and not a.output.exists();a.output.mkdir(parents=True)
assert not a.gradient_breakdown or a.baseline_only
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
 global source_state
 m=env['m']
 if a.initial_model:
  assert a.checkpoint.resolve()==Path(argv[argv.index('--parent')+1]).resolve()
  source_state={key:value.detach().cpu().clone() for key,value in m.state_dict().items()}
  artifact['optimizer']=copy.deepcopy(env['opt'].state_dict())
 else:
  m.load_state_dict(source_state);env['opt'].load_state_dict(copy.deepcopy(artifact['optimizer']))
 assert len(list(m.net.parameters()))==8
 for key,value in m.state_dict().items():assert torch.equal(value.cpu(),source_state[key])
from supermimic_motion_action_response import MotionActionResponseLoss
from supermimic_linear_dense_center import LinearDenseCenterLoss
_original_initialize=initialize
def initialize():
 _original_initialize()
 r=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics')
 env['_dynamic']=MotionActionResponseLoss(r/'motion_action_response_v1/serial_full',r/'motion_action_response_v1/full_integrity.json',env['cfg'])
 env['_center']=LinearDenseCenterLoss(r/'grounded_dynamic_cop_clarabel_init_v1/dense_native_collection_v1',env['cfg'],1)
def dot(x,y):return float(sum((u.double()*v.double()).sum() for u,v in zip(x,y)))
def population():
 data=env['_center'].data;model=env['response_model'];norm=model.state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std);sums={k:0. for k in ['linear','angular','joint']}
 with torch.no_grad():
  for start in range(0,895,64):
   stop=min(start+64,895);pred=model(data['state'][start:stop],data['action'][start:stop],data['context'][start:stop]);error=(pred-data['next_state'][start:stop])/std
   for k,sl in [('linear',slice(7,10)),('angular',slice(10,13)),('joint',slice(43,73))]:sums[k]+=float(.5*error[:,sl].square().mean())*(stop-start)/895
 return sums

def audit():
 assert a.baseline_only and a.gradient_breakdown
 params=list(env['m'].net.parameters());old=env['loss'];dynamic=env['_dynamic'](env['response_model']);center=env['_center'];idx=torch.randint(895,(16,),device='cuda',generator=center.generator)
 data=center.data;pred=env['response_model'](data['state'][idx],data['action'][idx],data['context'][idx]);norm=env['response_model'].state_norm;std=(norm.var+norm.eps).sqrt();std=torch.where(norm.frozen_mask,torch.ones_like(std),std);err=(pred-data['next_state'][idx])/std
 values={k:env[k] for k in ['rollout','h1','response_value','root_state_value','root_pose_value','velocity_anchor_value','motion_linear_value','native_value']}
 values.update(old_full=old,dynamic_response=dynamic,center_linear=.5*err[:,7:10].square().mean(),center_angular=.5*err[:,10:13].square().mean(),center_joint=.5*err[:,43:73].square().mean())
 values['new_full']=old+dynamic+values['center_linear']+values['center_angular']+values['center_joint']
 values['actual_full']=old+dynamic+values['center_angular']+values['center_joint']+int(artifact['motion_center_linear_enabled'])*values['center_linear']
 grads={k:torch.autograd.grad(v,params,retain_graph=True) for k,v in values.items()};g=grads['center_linear'];ng=dot(g,g)
 row={'batch':env['update'],'directions':{},'losses':{k:float(v) for k,v in values.items()},'center_indices':idx.cpu().tolist(),'dynamic_indices':env['_dynamic'].last_indices,'old_rng_sha256':[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for _,s in rng_before],'linear_pairs':{}}
 for k,v in grads.items():
  nv=dot(v,v);d=dot(g,v);row['linear_pairs'][k]={'dot':d,'cosine':d/max((ng*nv)**.5,1e-300),'gradient_norm':nv**.5}
 before=[v.detach().clone() for v in params];row['adam_probes']={};pop_before=population();row['population_before']=pop_before
 with torch.no_grad():row['static_before']=float(env['evaluate_guard_static']())
 for mode in ['saved_moments','shared_hidden','output_head','joint_velocity_head','all_first_moments']:
  shadow=[torch.nn.Parameter(v.clone()) for v in before];group=artifact['optimizer']['param_groups'][0];opt=torch.optim.Adam(shadow,lr=group['lr']);opt.load_state_dict(copy.deepcopy(artifact['optimizer']))
  assert len(group['params'])==8 and shadow[-2].shape[0]==72
  for i,pid in enumerate(group['params']):
   moment=opt.state[shadow[i]]['exp_avg']
   if mode=='all_first_moments' or (mode=='shared_hidden' and i<6) or (mode=='output_head' and i>=6):moment.zero_()
   elif mode=='joint_velocity_head' and i>=6:moment[39:69].zero_()
  for parameter,g in zip(shadow,grads['actual_full']):parameter.grad=g.detach().clone()
  torch.nn.utils.clip_grad_norm_(shadow,env['cfg']['wm_grad_clip']);opt.step();delta=[v.detach()-b for v,b in zip(shadow,before)]
  probe={'slopes':{k:dot(g,delta) for k,g in grads.items()},'joint_slope_by_layer':{'shared_hidden':dot(grads['center_joint'][:6],delta[:6]),'output_head':dot(grads['center_joint'][6:],delta[6:])}}
  try:
   with torch.no_grad():
    for parameter,value in zip(params,shadow):parameter.copy_(value)
    pred=env['response_model'](data['state'][idx],data['action'][idx],data['context'][idx]);error=(pred-data['next_state'][idx])/std
    probe['center_linear_after']=float(.5*error[:,7:10].square().mean());probe['center_linear_before']=float(values['center_linear'])
    probe['static_after']=float(env['evaluate_guard_static']());probe['population_after']=population()
  finally:
   with torch.no_grad():
    for parameter,value in zip(params,before):parameter.copy_(value)
  row['adam_probes'][mode]=probe
 assert all(torch.equal(p,b) for p,b in zip(params,before));assert population()==pop_before
 rows.append(row);(a.output/'partial.json').write_text(json.dumps(rows,indent=2));print('AUDITED',env['update'],flush=True)
def capture():
 global rng_before
 generators=[v for v in env.values() if isinstance(v,torch.Generator)]
 generators += [env[k].generator for k in ['response_loss','root_state_loss','root_pose_loss','motion_root_loss'] if env.get(k) is not None]
 assert len({id(g) for g in generators})==len(generators)
 rng_before=[(g,g.get_state().clone()) for g in generators]
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
plan=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(a.checkpoint),checkpoint_sha256=sha,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),batches=a.batches,baseline_only=a.baseline_only,gradient_breakdown=a.gradient_breakdown,initial_model=a.initial_model,cross_batch=a.cross_batch,two_batch_direction=a.two_batch_direction,two_batch_static=a.two_batch_static,common_descent=a.common_descent,component_losses=a.component_losses,response_common_descent=a.response_common_descent,adam_metric_common=a.adam_metric_common,all_component_common=a.all_component_common,project_full_descent=a.project_full_descent,argv=argv,scope='All actual fit losses/samplers at frozen initial or final weights; shadow Adam only, no source optimizer steps. Reset-first-moment retains second moments and step count; hypothetical diagnosis, not proposed accepted optimizer. Finite probes measure same fit-static batch only. Existing setup may load reference metadata; no validation loss used. Not a historical reconstruction of the 745 zero steps or closed-loop proof.')
(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
sys.argv=[str(source)]+argv;sys.path.insert(0,str(source.parent.resolve()))
env.update(__name__='__main__',__file__=str(source),_initialize=initialize,_audit=audit,_capture=capture)
exec(compile(program,str(source),'exec'),env)
for key,value in env['m'].state_dict().items():assert torch.equal(value.cpu(),source_state[key])
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha
summary={name:dict(static_ascent=sum(r['directions'][name]['static_slope']>0 for r in rows),full_ascent=sum(r['directions'][name]['full_slope']>0 for r in rows),identity_only=sum(not any(p['nonincrease'] for p in r['directions'][name]['probes'][:-1]) for r in rows),both_nonzero=sum(any(p['both_nonincrease'] for p in r['directions'][name]['probes'][:-1]) for r in rows) if a.common_descent else None) for name in rows[0]['directions']}
(a.output/'audit.json').write_text(json.dumps(dict(plan=plan,rows=rows,summary=summary,source_weights_unchanged=True,source_optimizer_steps=0,policy_acceptance=False),indent=2))
print(json.dumps(summary),flush=True)
