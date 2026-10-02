"""Exercise stateful target limiting without starting Isaac Sim."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch

def load_action():
    import torch
    class Base:
        def __init__(self,cfg,env):
            self.cfg=cfg;self._asset=env.asset;self.num_envs=2;self.device='cpu';self._joint_ids=[1,0]
            self._raw_actions=torch.zeros((2,2));self._processed_actions=self._raw_actions.clone()
        def process_actions(self,x):self._raw_actions=x.clone();self._processed_actions=x.clone()
        def reset(self,ids):self._raw_actions[ids]=0
    modules={}
    for name in ['isaaclab','isaaclab.envs','isaaclab.envs.mdp','isaaclab.envs.mdp.actions','isaaclab.envs.mdp.actions.joint_actions','isaaclab.envs.mdp.actions.actions_cfg','isaaclab.utils']:
        modules[name]=ModuleType(name)
    modules['isaaclab.envs.mdp.actions.joint_actions'].JointPositionAction=Base
    modules['isaaclab.envs.mdp.actions.actions_cfg'].JointPositionActionCfg=object
    modules['isaaclab.utils'].configclass=lambda x:x
    path=Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/smooth_joint_actions.py'
    spec=importlib.util.spec_from_file_location('_isolated_smooth_action',path);module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,modules):spec.loader.exec_module(module)
    return module.SmoothJointPositionAction

def test_rate_limit_joint_order_and_partial_reset():
    import torch
    cls=load_action();data=NS(joint_pos=torch.tensor([[1.,2.],[3.,4.]]),joint_vel_limits=torch.tensor([[2.,3.],[2.,3.]]))
    cfg=NS(smoothing_tau=0.,limit_target_velocity=True,target_velocity_scale=.5)
    term=cls(cfg,NS(step_dt=.02,asset=NS(data=data)))
    term.process_actions(torch.full((2,2),100.))
    torch.testing.assert_close(term._processed_actions,torch.tensor([[2.03,1.02],[4.03,3.02]]))
    term.reset([0]);data.joint_pos[0]=torch.tensor([10.,20.])
    term.process_actions(torch.full((2,2),100.))
    torch.testing.assert_close(term._processed_actions,torch.tensor([[20.03,10.02],[4.06,3.04]]))

def test_disabled_target_shaping_preserves_original_action():
    import torch
    cls=load_action();data=NS(joint_pos=torch.zeros((2,2)))
    term=cls(NS(smoothing_tau=0.,limit_target_velocity=False,target_velocity_scale=1.),NS(step_dt=.02,asset=NS(data=data)))
    actions=torch.tensor([[100.,-100.],[.1,.2]]);term.process_actions(actions)
    torch.testing.assert_close(term._processed_actions,actions)
