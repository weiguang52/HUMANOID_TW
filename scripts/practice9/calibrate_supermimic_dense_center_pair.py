"""Full dynamic-response objective plus a single exact action-center coefficient0/1."""
import argparse,ast,hashlib,json,shlex,sys
from pathlib import Path
from supermimic_motion_action_response import MotionActionResponseLoss
from supermimic_motion_action_center import MotionActionCenterLoss
from supermimic_dense_motion_center import DenseMotionCenterLoss
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--enabled',type=int,choices=[0,1],required=True);p.add_argument('--updates',type=int,choices=[4,1000],required=True);a=p.parse_args();assert not a.output.exists()
source=Path('scripts/practice9/calibrate_supermimic_branch_pair.py');runner=Path('scripts/practice9/run_supermimic_static_step_guard_pair.sh');command=next(x for x in runner.read_text().splitlines() if x.startswith('python -u '));argv=shlex.split(command.split(' > ',1)[0])[3:]
for k,v in [('--output',str(a.output)),('--updates',str(a.updates)),('--static-step-guard','0')]:argv[argv.index(k)+1]=v
E={};history=[];root=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_action_response_v1');dynamic=None;center=None;center_history=[]
def initialize():
 global dynamic,center
 dynamic=MotionActionResponseLoss(root/'serial_full',root/'full_integrity.json',E['cfg']);sparse=MotionActionCenterLoss(root/'serial_full',root/'full_integrity.json',E['cfg']);dense=DenseMotionCenterLoss(root.parent/'grounded_dynamic_cop_clarabel_init_v1/dense_native_collection_v1',E['cfg']);center=dense if a.enabled else sparse;plan=E['plan'];assert plan['base_prediction_endpoints_per_update']==784 and plan['prediction_endpoints_per_update']==2064
 plan.update(motion_action_response_enabled=True,motion_action_center_enabled=True,motion_center_dense_enabled=bool(a.enabled),motion_action_center_input=sparse.meta,motion_dense_center_input=dense.meta,motion_action_response_input=dynamic.meta,motion_action_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_prediction_endpoints_per_update=864,prediction_endpoints_per_update=2144,motion_action_rule='Keep complete old objective/static action response; both arms add64 dynamic response plus16 exact fit center endpoints; dynamic response1 and center1; sparse48 versus dense895 fit coverage only; same original Adam and disabled static observer guard.')
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
def extra():
 value=dynamic(E['response_model']);E['loss']=E['loss']+value;history.append({'update':E['update'],'indices':dynamic.last_indices,'value':float(value)})
 value=center(E['response_model']);E['loss']=E['loss']+value;center_history.append({'update':E['update'],'indices':center.last_indices,'value':float(value)})

def annotate(artifact):
 artifact['motion_action_response_enabled']=True;artifact['motion_action_center_enabled']=True;artifact['motion_center_dense_enabled']=bool(a.enabled);(a.output/'motion_action_center_history.json').write_text(json.dumps(center_history,indent=2));artifact['optimizer_update_contract']='original_adam_with_dynamic_action_response_v1';(a.output/'motion_action_history.json').write_text(json.dumps(history,indent=2))
tree=ast.parse(source.read_text());trial=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(z,ast.For) for z in n.body));loop=trial.body[0];cut=next(i for i,n in enumerate(loop.body) if i>0 and isinstance(n,ast.If) and ast.unparse(n.test)=='point_tally is not None');loop.body[cut:cut]=ast.parse('_extra()').body
for i,n in enumerate(trial.body):
 if isinstance(n,ast.Expr) and ast.unparse(n).startswith('torch.save(artifact,'):trial.body[i:i]=ast.parse('_annotate(artifact)').body;break
else:raise AssertionError('Missing artifact export')
for node in ast.walk(loop):
 if isinstance(node,ast.Assert) and 'point_tally' in ast.unparse(node):
  for value in ast.walk(node):
   if isinstance(value,ast.Constant) and value.value in (784,2064):value.value+=80
pos=tree.body.index(trial);tree.body[pos:pos]=ast.parse('_initialize()').body;ast.fix_missing_locations(tree);sys.argv=[str(source)]+argv;sys.path.insert(0,str(source.parent.resolve()));E.update(__name__='__main__',__file__=str(source),_extra=extra,_initialize=initialize,_annotate=annotate);exec(compile(tree,str(source),'exec'),E)
