"""Preserve failed GPU bitwise tests and distinguish source semantics from replay."""
import argparse,ast,hashlib,json
from pathlib import Path
import torch
from audit_supermimic_real_ppo_artifact import audit
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists();r=a.root;s=r/'real_ppo_multiseed_standing_v1'
old_path=s/'legacy_train_seed42.py';new_path=Path('scripts/practice9/train_supermimic_real_ppo.py')
class Default42(ast.NodeTransformer):
 def visit_Attribute(self,node):
  if isinstance(node.value,ast.Name) and node.value.id=='a' and node.attr=='seed':return ast.copy_location(ast.Constant(42),node)
  return self.generic_visit(node)
 def visit_Expr(self,node):
  call=node.value
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument' and call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value=='--seed':
   assert any(k.arg=='default' and isinstance(k.value,ast.Constant) and k.value.value==42 for k in call.keywords)
   return None
  return self.generic_visit(node)
 def visit_Assert(self,node):
  if ast.unparse(node.test)=='0 <= a.seed < 2 ** 31':return None
  return self.generic_visit(node)
old_tree=ast.parse(old_path.read_text());new_tree=Default42().visit(ast.parse(new_path.read_text()));ast.fix_missing_locations(new_tree)
assert ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False)
parent=Path('/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt');manifest=r.parent/'static_probe/manifest.json'
reports=[];models={}
for label,seed in [('legacy42',42),('legacy42_repeat',42),('default42',42),('seed43',43),('smooth43',43)]:
 reports.append(dict(label=label,audit=audit(s/'smoke'/label,parent,manifest,4,8,seed)))
 models[label]=torch.load(s/'smoke'/label/'checkpoint_final.pt',map_location='cpu',weights_only=False)
comparisons=[]
for label in ['legacy42_repeat','default42']:
 x,y=models['legacy42'],models[label]
 comparisons.append(dict(reference='legacy42',comparison=label,plan_equal=x['plan']==y['plan'],rng_equal=torch.equal(x['torch_rng_state'],y['torch_rng_state']),actor_bit_identical=all(torch.equal(v,y['actor_critic'][k]) for k,v in x['actor_critic'].items()),max_actor_tensor_difference=max(float((v-y['actor_critic'][k]).abs().max()) for k,v in x['actor_critic'].items() if v.dtype.is_floating_point)))
reused=[]
for arm,relative,coefficient,loss in [('baseline','real_ppo_standing_v1',0.,'squared'),('smooth','sole_distance_ppo_v1/trained',2.,'smooth_distance')]:
 run=r/relative;report=audit(run,parent,manifest,1000,256,42)
 c=torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False);assert c['plan'].get('sole_normal_penalty',0.)==coefficient and c['plan'].get('sole_normal_loss','squared')==loss
 reused.append(dict(arm=arm,seed=42,path=str(run),sha256=hashlib.sha256((run/'checkpoint_final.pt').read_bytes()).hexdigest(),audit=report))
out=dict(default42_source_ast_equivalence_passed=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old_path,new_path]},runtime_exact_regression_passed=all(x['actor_bit_identical'] for x in comparisons),comparisons=comparisons,artifact_audits=reports,reused_seed42=reused,scope='AST equality proves default42 source operation equivalence after substituting the declared default and removing only CLI declaration/range validation. It does not turn failed native GPU bitwise replay into a pass. No threshold adjustment, policy acceptance or formal training launch.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['artifact_audits','reused_seed42','source_sha256']}))
