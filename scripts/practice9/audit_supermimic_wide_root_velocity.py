"""Retain native root branch checks and regress all original velocity levels."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
source=Path(__file__).with_name('audit_supermimic_motion_root_velocity.py');tree=ast.parse(source.read_text());counts={'levels':0,'loop':0,'repeat':0,'size':0}
levels=[1.,2.,8.,32.,64.]
for n in ast.walk(tree):
 if isinstance(n,ast.Assert) and "plan['levels']" in ast.unparse(n.test):
  assert isinstance(n.test,ast.BoolOp);cmp=n.test.values[0];cmp.comparators[0]=ast.parse(repr(levels),mode='eval').body;counts['levels']+=1
 if isinstance(n,ast.For) and ast.unparse(n.target)=='(level, assignment)':n.iter=ast.parse(repr([(0.,0)]+[(l,i) for l in levels for i in [0,1]]),mode='eval').body;counts['loop']+=1
 if isinstance(n,ast.For) and ast.unparse(n.target)=='level':n.iter=ast.parse(repr(levels),mode='eval').body;counts['repeat']+=1
 if isinstance(n,ast.Assert) and 'len(expected)' in ast.unparse(n.test) and '480' in ast.unparse(n.test):
  for v in ast.walk(n):
   if isinstance(v,ast.Constant) and type(v.value)==int and v.value in [5,480]:v.value={5:11,480:1056}[v.value]
  counts['size']+=1
assert all(v==1 for v in counts.values()),counts
requested_output=Path(sys.argv[sys.argv.index('--output')+1]);assert not requested_output.exists();sys.argv[sys.argv.index('--output')+1]=str(requested_output.with_name(requested_output.stem+'.base_checks.json'))
env={'__name__':'__main__','__file__':str(source)};exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),env)
r=env['r'];plan=env['plan'];assert plan['contract']=='wide_motion_root_velocity_branches_v1'
variant=json.loads((r/'wide_velocity_diagnostic.json').read_text());assert variant['runtime_dense_verified'] and variant['levels']==levels
baseline=r.parent.parent/'grounded_motion_response_alignment_v1/serial_full';assert baseline.exists()
old={x['file']:x for x in json.loads((baseline/'results.json').read_text())['rows']};compared=[]
for row in env['result']['rows']:
 if row['level']>2:continue
 ref=old[row['file']];assert hashlib.sha256((baseline/ref['file']).read_bytes()).hexdigest()==ref['sha256']
 with np.load(r/row['file']) as a,np.load(baseline/ref['file']) as b:
  errors={k:float(np.abs(a[k]-b[k]).max()) for k in ['state','action','context','next_state']}
  assert errors['state']<=2e-6 and errors['action']==0 and errors['context']<=1e-7 and errors['next_state']<=1e-5,errors
  for k in ['terminated','truncated','native_qpos','native_qvel','native_qacc_warmstart']:np.testing.assert_array_equal(a[k],b[k])
 compared.append({'file':row['file'],'maximum_errors':errors})
assert len(compared)==(5 if plan['smoke'] else 480)
out=env['out'];out.update(original_levels_regression=compared,wide_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wide_levels=levels)
assert out['passed'],out['counts'];requested_output.write_text(json.dumps(out,indent=2));print('WIDE AUDIT PASSED; original batches retained',len(compared),flush=True)
