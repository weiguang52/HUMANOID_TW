"""Versioned complete-objective AST reuse for matched nine-loss common-gradient experiments."""
import argparse,ast,copy,hashlib,json,shlex,sys
from pathlib import Path
import torch
from supermimic_two_batch_step import two_batch_step
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--enabled',type=int,choices=[0,1],required=True);p.add_argument('--updates',type=int,choices=[4,1000],required=True);a=p.parse_args();assert not a.output.exists()
source=Path('scripts/practice9/calibrate_supermimic_branch_pair.py');runner=Path('scripts/practice9/run_supermimic_static_step_guard_pair.sh')
command=next(x for x in runner.read_text().splitlines() if x.startswith('python -u '));argv=shlex.split(command.split(' > ',1)[0])[3:]
for k,v in [('--output',str(a.output)),('--updates',str(a.updates)),('--static-step-guard','0')]:argv[argv.index(k)+1]=v
E={};history=[];POINTS=21056
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
def initialize():
 plan=E['plan'];assert plan['base_prediction_endpoints_per_update']==784
 plan.update(two_direction_step=bool(a.enabled),all_component_source_sha256=source_sha,prediction_endpoints_per_update=POINTS,guard_prediction_endpoints_per_update=POINTS-784,static_step_guard=False,static_step_guard_rule='Superseded by response_common_step contract',two_direction_rule='Two-batch full18 common direction, first admissible of fixed eight factors; identity if none admissible. Same full-gradient shadow Adam states, matched 21056 points both arms; no validation selection')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
def capture():
 global rng_before
 generators=[v for v in E.values() if isinstance(v,torch.Generator)]+[E[k].generator for k in ['response_loss','root_state_loss','root_pose_loss','motion_root_loss'] if E.get(k) is not None]
 assert len({id(g) for g in generators})==len(generators)
 rng_before=[(g,g.get_state().clone()) for g in generators]
def replay(states, gradient=False):
 saved=[(g,g.get_state().clone()) for g,_ in rng_before];sandbox=E.copy()
 for k,v in sandbox.items():
  if isinstance(v,list) and k.endswith('history'):sandbox[k]=list(v)
 sandbox['point_tally']=dict(E['point_tally'])
 try:
  for g,state in states:g.set_state(state)
  with torch.set_grad_enabled(gradient):
   exec(replay_program,sandbox)
   batch={k:v[:32] for k,v in sandbox['rollout_batch'].items()}
   state=batch['state'];static=0
   for j in range(8):
    state=sandbox['predictor'](state,batch['action'][:,j],batch['context'][:,j]);target=batch['next_state'][:,j]
    static=static+(sandbox['m'].state_loss(state,target)+sandbox['velocity_value_loss'](state,target))/8
   values=(static,sandbox['loss'],*[sandbox[k] for k in ['response_value','root_state_value','root_pose_value','rollout','h1','velocity_anchor_value','motion_linear_value']])
   if gradient:
    params=list(E['m'].net.parameters())
    grads=[torch.autograd.grad(v,params,retain_graph=i<len(values)-1) for i,v in enumerate(values)]
    return tuple(float(v) for v in values),grads
   return values
 finally:
  for g,state in saved:g.set_state(state)
  assert all(torch.equal(g.get_state(),state) for g,state in saved)

def step():
 rng_cross=[(g,g.get_state().clone()) for g,_ in rng_before]
 params=list(E['m'].net.parameters());static=E['evaluate_guard_static']();gs=torch.autograd.grad(static,params);gf=torch.autograd.grad(E['loss'],params,retain_graph=True)
 gr=[torch.autograd.grad(E[key],params,retain_graph=True) for key in ['response_value','root_state_value','root_pose_value','rollout','h1','velocity_anchor_value','motion_linear_value']]
 baseline=(float(static),float(E['loss']),*[float(E[key]) for key in ['response_value','root_state_value','root_pose_value','rollout','h1','velocity_anchor_value','motion_linear_value']])
 assert baseline[0]==E['guard_baseline']
 cross_baseline,cross_gradients=replay(rng_cross,True)
 baseline=baseline+cross_baseline
 def evaluate():return (*replay(rng_before),*replay(rng_cross))
 proof=two_batch_step(params,E['opt'],[gs,gf,*gr,*cross_gradients],evaluate,baseline,E['cfg']['wm_grad_clip'],bool(a.enabled))
 assert E['point_tally']['points']==POINTS,(E['point_tally'],POINTS)
 E['point_history'].append(POINTS);history.append(dict(update=E['update'],static_indices=E['idx'][:32].cpu().tolist(),first_rng_sha256=[hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest() for _,state in rng_before],second_rng_sha256=[hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest() for _,state in rng_cross],**proof))
 (a.output/'two_direction_history.json').write_text(json.dumps(history,indent=2))
 if E['update']%100==0 or E['update']==a.updates:print(json.dumps(dict(update=E['update'],loss=baseline[1],factor=proof['selected_factor'])),flush=True)
def annotate(artifact):
 artifact['optimizer_update_contract']='two_direction_fit_finite_step_full_adam_state_v1' if a.enabled else 'unmodified_adam_with_two_direction_observer_v1'
 artifact['two_direction_history_updates']=len(history)
tree=ast.parse(source.read_text());trial=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(z,ast.For) for z in n.body));loop=trial.body[0]
assert isinstance(loop,ast.For) and ast.unparse(loop.target)=='update'
replay_cut=next(i for i,n in enumerate(loop.body) if i>0 and isinstance(n,ast.If) and ast.unparse(n.test)=='point_tally is not None')
replay_program=compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(loop.body[:replay_cut]),type_ignores=[])),str(source),'exec')
cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='loss.backward()')
loop.body=ast.parse('_capture()').body+loop.body[:cut]+ast.parse('_step()').body
# Preserve original failure/checkpoint export/frozen-source assertions.
for i,n in enumerate(trial.body):
 if isinstance(n,ast.Expr) and ast.unparse(n).startswith('torch.save(artifact,'):
  trial.body[i:i]=ast.parse('_annotate(artifact)').body;break
else:raise AssertionError('Expected final artifact export')
pos=tree.body.index(trial);tree.body[pos:pos]=ast.parse('_initialize()').body;ast.fix_missing_locations(tree)
sys.argv=[str(source)]+argv;sys.path.insert(0,str(source.parent.resolve()));E.update(__name__='__main__',__file__=str(source),_capture=capture,_step=step,_initialize=initialize,_annotate=annotate)
exec(compile(tree,str(source),'exec'),E)
