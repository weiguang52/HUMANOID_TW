import importlib.util
from pathlib import Path
import pytest
import torch

path = Path(__file__).parents[2] / 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/support_velocity.py'
spec = importlib.util.spec_from_file_location('support_velocity', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def cost(reference=None, actual=None, yaw=None, standing=None, contact=None, known=None, **kwargs):
    return m.support_velocity_cost(
        torch.zeros(1, 2) if reference is None else torch.tensor(reference),
        torch.tensor([[0.02, 0.]]) if actual is None else torch.tensor(actual),
        torch.zeros(1) if yaw is None else torch.tensor(yaw),
        torch.tensor([True]) if standing is None else torch.tensor(standing),
        torch.tensor([[True, True]]) if contact is None else torch.tensor(contact),
        torch.tensor([[True, True]]) if known is None else torch.tensor(known), **kwargs)


def test_zero_when_tracking_and_positive_when_drifting():
    assert cost(actual=[[0., 0.]]).item() == 0
    assert 0 < cost().item() < 1
    assert cost(actual=[[0.04, 0.]]).item() > cost().item()


def test_contact_unknown_and_nonstanding_cannot_activate():
    assert cost(standing=[False]).item() == 0
    assert cost(known=[[True, False]]).item() == 0
    assert cost(contact=[[True, False]]).item() == 0


def test_reference_motion_and_turns_disable_penalty():
    assert cost(reference=[[0.025, 0.]]).item() == 0
    assert cost(yaw=[0.3]).item() == 0
    assert cost(yaw=[-0.3]).item() == 0
    assert 0 < cost(yaw=[0.2]).item() < cost().item()


def test_small_intentional_reference_velocity_is_tracked():
    assert cost(reference=[[0.01, 0.]], actual=[[0.01, 0.]]).item() == 0
    assert cost(reference=[[0.01, 0.]], actual=[[0., 0.]]).item() > 0


def test_horizontal_rotation_invariance_and_finite_gradient():
    ref = torch.tensor([[0.003, 0.004]])
    actual = torch.tensor([[0.02, -0.01]], requires_grad=True)
    args = (torch.zeros(1), torch.ones(1, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    value = m.support_velocity_cost(ref, actual, *args)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    torch.testing.assert_close(value, m.support_velocity_cost(ref @ rotation, actual @ rotation, *args))
    value.sum().backward()
    assert torch.isfinite(actual.grad).all()


@pytest.mark.parametrize('kwargs', [{'scale': 0}, {'scale': float('nan')},
    {'speed_low': .03}, {'yaw_low': .4}, {'speed_low': -.1}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        cost(**kwargs)
