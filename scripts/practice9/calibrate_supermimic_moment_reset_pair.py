"""Matched static-guard experiment: reset only Adam's first moment before each step."""
import argparse,ast,hashlib,json,shlex,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--enabled',type=int,choices=[0,1],required=True)
p.add_argument('--updates',type=int,choices=[4,1000],required=True)
a=p.parse_args();assert not a.output.exists()
source=Path('scripts/practice9/calibrate_supermimic_branch_pair.py')
runner=Path('scripts/practice9/run_supermimic_static_step_guard_pair.sh')
command=next(x for x in runner.read_text().splitlines() if x.startswith('python -u '))
argv=shlex.split(command.split(' > ',1)[0])[3:]
for key,value in [('--output',str(a.output)),('--updates',str(a.updates)),('--static-step-guard','1')]:argv[argv.index(key)+1]=value
E={};resets=[]
def step():
 opt=E['opt'];count=0
 if a.enabled:
  for state in opt.state.values():
   if 'exp_avg' in state:state['exp_avg'].zero_();count+=1
 opt.step();resets.append(dict(update=E['update'],reset_tensors=count))
def annotate(artifact):
 artifact['optimizer_update_contract']='static_guard_first_moment_reset_v1' if a.enabled else 'static_guard_original_adam_v1'
 artifact['first_moment_reset_enabled']=bool(a.enabled)
 (a.output/'moment_reset_history.json').write_text(json.dumps(resets,indent=2))
def initialize():
 E['plan'].update(first_moment_reset_enabled=bool(a.enabled),first_moment_reset_rule='Clear exp_avg immediately before each real Adam step; retain exp_avg_sq, step, learning rate, gradients, clipping, original finite static guard and all samplers.',moment_reset_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (a.output/'plan.json').write_text(json.dumps(E['plan'],indent=2))
tree=ast.parse(source.read_text());trial=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(z,ast.For) for z in n.body));loop=trial.body[0]
steps=[n for n in ast.walk(loop) if isinstance(n,ast.Expr) and ast.unparse(n)=='opt.step()'];assert len(steps)==1
steps[0].value=ast.Call(func=ast.Name(id='_moment_step',ctx=ast.Load()),args=[],keywords=[])
for i,n in enumerate(trial.body):
 if isinstance(n,ast.Expr) and ast.unparse(n).startswith('torch.save(artifact,'):
  trial.body[i:i]=ast.parse('_annotate(artifact)').body;break
else:raise AssertionError('Missing artifact export')
pos=tree.body.index(trial);tree.body[pos:pos]=ast.parse('_initialize()').body;ast.fix_missing_locations(tree)
sys.argv=[str(source)]+argv;sys.path.insert(0,str(source.parent.resolve()))
E.update(__name__='__main__',__file__=str(source),_moment_step=step,_initialize=initialize,_annotate=annotate)
exec(compile(tree,str(source),'exec'),E)
