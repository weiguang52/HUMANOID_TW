from pathlib import Path
import sys,json,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
from reference_speed_blend import mix_actions
from candidate_action_scope import make_scope_contract,mix_scoped_actions
names=json.loads(Path('/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/videos/validation/manifest.json').read_text())['joint_names']
device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
if device not in ['cpu','cuda'] or (device=='cuda' and not torch.cuda.is_available()):raise ValueError('Unavailable test device')
c=make_scope_contract(names,'lower_body_waist');assert len(c['candidate_joint_names'])==17 and len(c['teacher_joint_names'])==13
mask=torch.tensor([n in c['candidate_joint_names'] for n in names],device=device);torch.manual_seed(42)
t=torch.randn(2,5,30,device=device);s=torch.randn_like(t,requires_grad=True);w=torch.tensor([[0,.2,.5,.9,1.],[1,.9,.5,.2,0.]],device=device)
t0=t.clone();s0=s.detach().clone();rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
for weight in [torch.zeros_like(w),torch.ones_like(w),w]:
 old=mix_actions(t,s,weight);default=mix_scoped_actions(t,s,weight)
 assert torch.equal(old,default)
 if bool((weight==0).all()):assert default is t
 if bool((weight==1).all()):assert default is s
 out=mix_scoped_actions(t,s,weight,'lower_body_waist',names)
 assert torch.equal(out[...,mask],old[...,mask]) and torch.equal(out[...,~mask],t[...,~mask])
assert torch.equal(t,t0) and torch.equal(s.detach(),s0) and torch.equal(rng,torch.random.get_rng_state())
if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
out=mix_scoped_actions(t,s,w,'lower_body_waist',names);out.sum().backward();expected=w[...,None].expand_as(s).clone();expected[...,~mask]=0;assert torch.equal(s.grad,expected)
assert torch.equal(mix_scoped_actions(t,s,w,'lower_body_waist',(n for n in names)),out)
order=list(reversed(range(30)));permuted=mix_scoped_actions(t[...,order],s.detach()[...,order],w,'lower_body_waist',[names[i] for i in order]);assert torch.equal(permuted,out.detach()[...,order])
for bad in [names[:-1],names+[names[0]],['unknown']+names[1:]]:
 try:make_scope_contract(bad,'lower_body_waist')
 except ValueError:pass
 else:raise AssertionError('bad joint mapping accepted')
for weight in [torch.full_like(w,float('nan')),w+1,torch.ones(2)]:
 try:mix_scoped_actions(t,s,weight,'lower_body_waist',names)
 except ValueError:pass
 else:raise AssertionError('bad weights accepted')
try:mix_scoped_actions(t,s,w,'unknown',names)
except ValueError:pass
else:raise AssertionError('unknown scope accepted')
print(device,'scope default bit-exact;17 student/13 teacher; gradient, RNG, permutation, shape and rejection checks passed')
print('Isolated helper only: no training/deployment integration or quality claim')
