import importlib.util
from pathlib import Path
import torch
import pytest

p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/size_tracking.py'
spec=importlib.util.spec_from_file_location('size_tracking',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_scale_invariance_and_zero():
    error=torch.tensor([[0.,0.,0.],[.04,.03,0.]])
    a=m.normalized_pseudo_huber(error,.02)
    b=m.normalized_pseudo_huber(error*.3,.006)
    assert torch.allclose(a,b)
    assert a[0]==0 and a[1]>0


def test_large_error_still_changes_cost():
    error=torch.tensor([[.2,0.,0.]],requires_grad=True)
    cost=m.normalized_pseudo_huber(error,.02)
    cost.sum().backward()
    assert torch.isfinite(error.grad).all() and error.grad[0,0]>40
    assert m.normalized_pseudo_huber(torch.tensor([[.1,0.,0.]]),.02)<cost


def test_invalid_scale():
    with pytest.raises(ValueError):m.normalized_pseudo_huber(torch.zeros(1,3),0)
