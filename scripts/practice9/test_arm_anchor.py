from pathlib import Path
import sys,math,json,torch,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from anchored_ppo import arm_anchor_weights,hip_anchor_weights,anchor_mse,ARM_ANCHOR_NAMES
names=json.loads(Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/videos/validation/manifest.json').read_text())['joint_names']
assert len(names)==30 and len(ARM_ANCHOR_NAMES)==10
ids=[names.index(n) for n in ARM_ANCHOR_NAMES]
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 torch.manual_seed(42);s=torch.randn(17,30,device=device,requires_grad=True);t=torch.randn_like(s)
 one=torch.tensor(arm_anchor_weights(names,1),device=device);assert torch.equal(anchor_mse(s,t),anchor_mse(s,t,one))
 w=torch.tensor(arm_anchor_weights(names,4),device=device);assert int((w==4).sum())==10 and int((w==1).sum())==20
 loss=anchor_mse(s,t,w);expected=(s-t).square().mean()+(s[:,ids]-t[:,ids]).square().mean()
 assert torch.allclose(loss,expected,rtol=1e-6,atol=1e-6)
 g=torch.autograd.grad(loss,s,retain_graph=True)[0];ge=torch.autograd.grad(expected,s)[0];assert torch.allclose(g,ge,rtol=1e-6,atol=1e-7)
 order=torch.randperm(30).tolist();permuted=arm_anchor_weights([names[i] for i in order],4);assert permuted==[float(w[i]) for i in order]
 hip=hip_anchor_weights(names,3);both=arm_anchor_weights(names,4,hip);assert all(both[i]==(4 if names[i] in ARM_ANCHOR_NAMES else hip[i]) for i in range(30))
 print('ARM_MATH_GRADIENT_DEFAULT_PERMUTATION_PASS',device)
for bad in [float('nan'),float('inf'),0,-1,.99]:
 try:arm_anchor_weights(names,bad)
 except ValueError:pass
 else:raise AssertionError(bad)
for badnames in [names[:-1],names+[names[0]],['unknown' if n==ARM_ANCHOR_NAMES[0] else n for n in names]]:
 try:arm_anchor_weights(badnames,4)
 except ValueError:pass
 else:raise AssertionError('invalid joint names accepted')
for existing in [[1]*29,[float('nan')]*30,[0]*30]:
 try:arm_anchor_weights(names,4,existing)
 except ValueError:pass
 else:raise AssertionError('invalid weights accepted')
p=Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/source_before_arm_anchor/anchored_ppo.py');old=ast.parse(p.read_text());new=ast.parse((Path(__file__).resolve().parents[1]/'rsl_rl/anchored_ppo.py').read_text());functions=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))};o=functions(old);n=functions(new);assert all(n[k]==v for k,v in o.items())
print('OLD_PPO_UPDATE_AND_ALL_OLD_HELPERS_AST_IDENTICAL')
print('ARM_INVALID_ARGUMENTS_PASS')

from anchored_ppo import validate_arm_resume
validate_arm_resume({},1);validate_arm_resume({},4);validate_arm_resume({'arm_multiplier':4},4)
for value in [1,3,float('nan')]:
 try:validate_arm_resume({'arm_multiplier':4},value)
 except ValueError:pass
 else:raise AssertionError('silent resume change accepted')
print('ARM_RESUME_PRESERVATION_PASS')
