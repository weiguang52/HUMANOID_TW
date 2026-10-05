import importlib,sys,types,math
from pathlib import Path
import torch
import pytest
pkg=types.ModuleType('tw154_mdp');pkg.__path__=[str(Path(__file__).resolve().parents[2]/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp')]
sys.modules['tw154_mdp']=pkg
path=importlib.import_module('tw154_mdp.path_tracking')
rate=importlib.import_module('tw154_mdp.target_rate_scales')


def test_yaw_frame_and_translation_invariance():
 q=torch.tensor([[math.sqrt(.5),0.,0.,math.sqrt(.5)]])
 assert torch.allclose(path.yaw_frame_xy(torch.tensor([[1.,0.]]),q),torch.tensor([[0.,-1.]]),atol=1e-6)
 c=types.SimpleNamespace(anchor_pos_w=torch.tensor([[10.,3.,.2]]),robot_anchor_pos_w=torch.tensor([[9.,3.,.2]]),robot_anchor_quat_w=q)
 env=types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name:c))
 before=path.path_error_observation(env,'motion',1.)
 c.anchor_pos_w+=100;c.robot_anchor_pos_w+=100
 assert torch.allclose(before,path.path_error_observation(env,'motion',1.))
 assert path.path_error_cost(env,'motion',1.)>0
 c.anchor_pos_w=c.robot_anchor_pos_w.clone()
 assert path.path_error_cost(env,'motion',1.)==0


def test_leg_rate_override_preserves_upper_body_and_order():
 assert rate.resolve_scales(['arm','knee','waist'],.5,{'knee':.75})==[.5,.75,.5]
 assert rate.resolve_scales(['a','b'],.5)==[.5,.5]
 with pytest.raises(ValueError):rate.resolve_scales(['a'],.5,{'b':.75})
 with pytest.raises(ValueError):rate.resolve_scales(['a'],.5,{'a':1.1})
