import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import torch

p=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/joint_compensation.py'
spec=importlib.util.spec_from_file_location('compensation',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_group_means_do_not_depend_on_joint_count_and_preserve_grad():
    q=torch.zeros((2,3))
    target=torch.tensor([[.25,.25,.5],[0.,0.,0.]],requires_grad=True)
    names=['a','b','c']
    robot=NS(data=NS(joint_pos=q,joint_pos_target=target),
        find_joints=lambda selected,preserve_order:([names.index(n) for n in selected],selected))
    env=NS(scene={'robot':robot})
    torch.testing.assert_close(m.group_target_error(env,['a']),torch.tensor([1.,0.]))
    torch.testing.assert_close(m.group_target_error(env,['a','b']),torch.tensor([1.,0.]))
    value=m.group_target_error(env,['c','a'])
    torch.testing.assert_close(value,torch.tensor([2.5,0.]))
    value.sum().backward()
    assert target.grad[0,1]==0 and target.grad[0,0]>0 and target.grad[0,2]>0
