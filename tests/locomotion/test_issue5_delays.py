"""Delay-buffer behavior tests; physics integration is verified by training smoke."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
import torch

class BufferBase:
    def __init__(self,cfg,env):self._raw_actions=torch.zeros((env.num_envs,2))
    def process_actions(self,actions):self._raw_actions.copy_(actions)
    def reset(self,env_ids=None):self._raw_actions[slice(None) if env_ids is None else env_ids]=0

def load_module():
    fake=types.ModuleType('isaaclab.envs.mdp.actions');fake.JointPositionAction=BufferBase
    key='isaaclab.envs.mdp.actions';old=sys.modules.get(key);sys.modules[key]=fake
    try:
        path=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/tw/experiments.py'
        spec=importlib.util.spec_from_file_location('issue5_delay_test_module',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    finally:
        if old is None:sys.modules.pop(key,None)
        else:sys.modules[key]=old

class DelayTests(unittest.TestCase):
    def test_zero_one_two_step_targets(self):
        cls=load_module().RandomTwoStepDelayAction
        a=cls(None,types.SimpleNamespace(num_envs=3,device='cpu'));a.lag[:]=torch.tensor([[0],[1],[2]])
        for value,expected in [(1,[1,0,0]),(2,[2,1,0]),(3,[3,2,1])]:
            a.process_actions(torch.full((3,2),float(value)))
            self.assertTrue(torch.equal(a._raw_actions[:,0],torch.tensor(expected,dtype=torch.float)))
        a.reset([1]);self.assertFalse(a.previous[1].any());self.assertFalse(a.older[1].any())
        self.assertTrue(torch.equal(a.previous[0],torch.tensor([3.,3.])))
    def test_fixed_delay_remains_one_after_resets(self):
        a=load_module().FixedOneStepDelayAction(None,types.SimpleNamespace(num_envs=4,device='cpu'))
        for ids in [None,[0,2],[1,3]]:
            a.reset(ids);self.assertTrue(a.delay.all())
        a.process_actions(torch.ones((4,2)));self.assertFalse(a._raw_actions.any())
        a.process_actions(torch.full((4,2),2.));self.assertTrue(torch.equal(a._raw_actions,torch.ones((4,2))))

if __name__=='__main__':unittest.main()
