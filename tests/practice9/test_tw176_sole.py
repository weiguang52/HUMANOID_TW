import importlib.util,pathlib
import torch,pytest
p=pathlib.Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/sole_tracking.py'
spec=importlib.util.spec_from_file_location('sole',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def box():
 import itertools
 return torch.tensor([[list(x) for x in itertools.product([-.1,.1],[-.2,.2],[-.03,.03])]])

def test_identity_translation_and_pitch():
 p=torch.tensor([[[0.,0.,.1]]]);q=torch.tensor([[[1.,0.,0.,0.]]])
 assert torch.allclose(s.sole_height(p,q,box()),torch.tensor([[.07]]))
 q=torch.tensor([[[2**-.5,0.,2**-.5,0.]]])
 assert torch.allclose(s.sole_height(p,q,box()),torch.tensor([[0.]]),atol=1e-6)

def test_batched_geometry():
 p=torch.zeros(3,1,3);q=torch.tensor([1.,0.,0.,0.]).expand(3,1,4)
 assert s.world_corners(p,q,box()).shape==(3,1,8,3)

def test_zero_cost_and_nonzero_large_error_gradient():
 x=torch.tensor([[.04,.01]],requires_grad=True)
 assert s.sole_cost(x,x,.01).item()==0
 s.sole_cost(torch.zeros_like(x),x,.01).backward();assert (x.grad>0).all()

@pytest.mark.parametrize('scale',[0,-1,float('nan')])
def test_bad_scale(scale):
 with pytest.raises(ValueError):s.sole_cost(torch.zeros(1,2),torch.zeros(1,2),scale)

def test_touchdown_matching_does_not_reuse_events():
 # Resolve repository root independently of module package depth.
 root=pathlib.Path(__file__).resolve().parents[2]
 spec=importlib.util.spec_from_file_location('diagnostic',root/'scripts/practice9/diagnose_sole_contacts.py')
 d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
 x=d.event_lags([100,110],[102],tolerance=.2)
 assert x['lags_s']==[.04] and x['unmatched_reference']==1
