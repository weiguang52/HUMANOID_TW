import sys,pathlib,types,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts/rsl_rl'))
from standing_diagnostic import reference_actions

def test_reference_action_inverse_with_reordered_joint_ids():
 robot=types.SimpleNamespace(joint_names=['a','b','c'])
 action=types.SimpleNamespace(_joint_ids=[2,0,1],_asset=robot,_scale=torch.tensor([[2.,.5,1.]]),_offset=torch.tensor([[.1,.2,.3]]))
 env=types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda name:action))
 command=types.SimpleNamespace(robot=robot,joint_indexes=[1,2,0],joint_pos=torch.tensor([[.6,.8,.4]]))
 raw=reference_actions(env,command)
 torch.testing.assert_close(raw*action._scale+action._offset,torch.tensor([[.8,.4,.6]]))

def test_zero_scale_rejected():
 import pytest
 robot=types.SimpleNamespace(joint_names=['a']);act=types.SimpleNamespace(_joint_ids=slice(None),_asset=robot,_scale=0.,_offset=0.)
 env=types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda _:act));cmd=types.SimpleNamespace(robot=robot,joint_indexes=[0],joint_pos=torch.ones(1,1))
 with pytest.raises(ValueError):reference_actions(env,cmd)
