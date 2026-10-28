from pathlib import Path
import sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from reference_speed_blend import candidate_weight,mix_actions
s=torch.tensor([0.,.04,.05,.075,.1,.2]);w=candidate_weight(s)
assert torch.allclose(w,torch.tensor([0.,0.,0.,.25,.5,.5]),atol=1e-7)
a=torch.randn(6,30);b=torch.randn(6,30)
assert mix_actions(a,b,torch.zeros(6)) is a
assert torch.equal(mix_actions(a,b,w)[:3],a[:3])
assert torch.equal(mix_actions(a,b,torch.ones(6)),b)
assert torch.isfinite(mix_actions(a,b,w)).all()
x=torch.linspace(0,.2,1001);g=candidate_weight(x);assert bool((g[1:]>=g[:-1]).all())
print('Gate boundaries, monotonicity, teacher identity, and action endpoints passed')
