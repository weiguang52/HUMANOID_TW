"""Differentiable penalty for imagined joint velocities beyond physical caps."""
import torch

def velocity_violation_cost(velocity: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared relative excess; states are not projected/clipped."""
    if velocity.shape[-1] != limits.numel():
        raise ValueError('Joint velocity/limit dimensions differ')
    return torch.relu(velocity.abs() / limits - 1.0).square().mean(dim=-1)
