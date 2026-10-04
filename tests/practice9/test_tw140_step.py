import importlib.util
import pathlib
import sys
import types
import torch

ROOT=pathlib.Path(__file__).resolve().parents[2]
pkg='tw140_test_mdp'
mod=types.ModuleType(pkg)
mod.__path__=[str(ROOT/'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp')]
sys.modules[pkg]=mod
s=importlib.import_module(pkg+'.step_tracking')


def test_preview_does_not_cross_clip_boundary():
    starts=torch.tensor([0,3])
    motion=types.SimpleNamespace(clip_lengths=torch.tensor([3,5]),
        frame_indices=lambda ids,local: starts[ids]+local)
    c=types.SimpleNamespace(motion=motion,motion_ids=torch.tensor([0,1]),
        time_steps=torch.tensor([2,3]))
    actual=s.preview_indices(c,[0,5,15])
    assert [x.tolist() for x in actual]==[[2,6],[2,7],[2,7]]


def test_swing_unknown_and_stance_have_no_reward_or_gradient():
    error=torch.tensor([[.0,.03],[.04,.0]],requires_grad=True)
    contact=torch.tensor([[False,False],[True,False]])
    known=torch.tensor([[True,False],[True,False]])
    reward=s.masked_swing_score(error,contact,known,.02)
    assert reward.tolist()==[1.,0.]
    reward.sum().backward()
    assert error.grad.abs().sum()==0
    displaced=s.masked_swing_score(torch.tensor([[.02,0.]]),
        torch.tensor([[False,True]]),torch.tensor([[True,True]]),.02)
    assert 0.36<float(displaced)<0.38


def test_preview_keeps_unknown_bits_and_uses_control_timestep():
    motion=types.SimpleNamespace(clip_lengths=torch.tensor([3]),
        frame_indices=lambda ids,local:local,
        joint_pos=torch.tensor([[1.],[2.],[3.]]))
    c=types.SimpleNamespace(motion=motion,motion_ids=torch.tensor([0]),
        time_steps=torch.tensor([0]),frame_indices=torch.tensor([0]),
        cfg=types.SimpleNamespace(joint_names=['knee']),
        _contact_reference=(torch.tensor([[True,False],[False,True],[True,True]]),
                            torch.tensor([[False,True],[True,False],[True,True]])))
    env=types.SimpleNamespace(step_dt=.02,
        command_manager=types.SimpleNamespace(get_term=lambda name:c))
    out=s.leg_preview(env,'motion',['knee'],horizons=(0.,.02,.3))
    assert out.tolist()==[[1.,1.,0.,0.,1.,2.,0.,1.,1.,0.,3.,1.,1.,1.,1.]]
