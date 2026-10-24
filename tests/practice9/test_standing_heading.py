import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/standing_heading.py'
s=importlib.util.spec_from_file_location('heading',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def q(angle):
 a=torch.as_tensor(angle);return torch.stack((torch.cos(a/2),torch.zeros_like(a),torch.zeros_like(a),torch.sin(a/2)),-1)
def cost(ref,actual,speed=0.,rate=0.,standing=True):
 return m.heading_cost(q([ref]),q([actual]),torch.tensor([[speed,0.]]),torch.tensor([rate]),torch.tensor([standing]))
def test_zero_and_wrapped_angle():
 assert cost(0.,0.).item()==0
 assert torch.allclose(cost(3.13,-3.13),cost(0.,0.023185307),atol=1e-6)
def test_quaternion_sign_and_world_yaw_invariance():
 a=q([0.3]);b=q([0.7]);args=(torch.zeros(1,2),torch.zeros(1),torch.ones(1,dtype=torch.bool))
 assert torch.allclose(m.heading_cost(a,b,*args),m.heading_cost(-a,b,*args))
 assert torch.allclose(cost(.3,.7),cost(1.3,1.7),atol=1e-6)
@pytest.mark.parametrize('speed,rate,standing',[(.025,0.,True),(0.,.3,True),(0.,0.,False)])
def test_excludes_moving_turning_and_other_categories(speed,rate,standing):
 assert cost(0.,1.,speed,rate,standing).item()==0

def test_gradient_does_not_saturate_and_is_finite():
 a=torch.tensor([1.],requires_grad=True)
 value=m.heading_cost(q([0.]),q(a),torch.zeros(1,2),torch.zeros(1),torch.ones(1,dtype=torch.bool))
 value.sum().backward();assert torch.isfinite(a.grad).all() and a.grad.item()>0

@pytest.mark.parametrize('scale',[0.,-1.,float('nan')])
def test_invalid_scale(scale):
 with pytest.raises(ValueError):m.heading_cost(q([0.]),q([0.]),torch.zeros(1,2),torch.zeros(1),torch.ones(1,dtype=torch.bool),scale)
