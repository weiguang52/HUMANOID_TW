"""Floating-base speed constraints must conserve momentum and remove energy."""

import torch
from supermimic.config.tw.velocity_projection import project_velocity_mass_metric
from supermimic.config.tw.velocity_limits import LIMITS


def test_coupled_root_reaction_matches_analytic_solution():
    mass = torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], dtype=torch.float64)
    velocity = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    actual, stats = project_velocity_mass_metric(
        velocity, mass, torch.tensor([0.0, 2.0]), free_dofs=1
    )
    torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert stats["momentum_error"] < 1e-12
    assert stats["energy_increase"] == 0.0


def test_full_30dof_projection_preserves_free_momentum_energy_and_bounds():
    torch.manual_seed(42)
    batch = 8
    transform = torch.eye(36, dtype=torch.float64).expand(batch, -1, -1).clone()
    transform[:, :6, 6:] = torch.randn(batch, 6, 30, dtype=torch.float64) * 0.05
    base = (
        torch.diag(
            torch.tensor(
                [0.8, 0.8, 0.8, 0.002, 0.002, 0.002] + [0.001] * 30, dtype=torch.float64
            )
        )
        .expand(batch, -1, -1)
        .clone()
    )
    a = torch.randn(batch, 30, 30, dtype=torch.float64) * 0.005
    base[:, 6:, 6:] += a.transpose(-1, -2) @ a
    mass = transform.transpose(-1, -2) @ base @ transform
    velocity = torch.randn(batch, 36, dtype=torch.float64) * 8
    caps = torch.tensor([0.0] * 6 + [2.0] * 30, dtype=torch.float64)
    actual, stats = project_velocity_mass_metric(velocity, mass, caps)
    assert (actual[:, 6:].abs() <= 2.0).all()
    before = (mass @ velocity.unsqueeze(-1)).squeeze(-1)
    after = (mass @ actual.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(after[:, :6], before[:, :6], atol=1e-10, rtol=1e-10)
    assert ((actual * after).sum(-1) <= (velocity * before).sum(-1) + 1e-10).all()
    naive = velocity.clone()
    naive[:, 6:] = naive[:, 6:].clamp(-2.0, 2.0)
    assert (
        (mass @ naive.unsqueeze(-1)).squeeze(-1)[:, :6] - before[:, :6]
    ).abs().max() > 0.001
    assert stats["residual"] <= 1e-5


def test_inactive_limits_leave_all_velocity_bits_unchanged():
    velocity = torch.tensor([[100.0, 0.1]], dtype=torch.float64)
    actual, stats = project_velocity_mass_metric(
        velocity,
        torch.eye(2, dtype=torch.float64)[None],
        torch.tensor([0.0, 2.0]),
        free_dofs=1,
    )
    assert actual is velocity
    assert not stats["active"]


def test_original_tw_joint_group_caps():
    assert len(LIMITS) == 30
    assert sum(cap == 3.0 for cap in LIMITS.values()) == 14
    assert sum(cap == 2.0 for cap in LIMITS.values()) == 16
