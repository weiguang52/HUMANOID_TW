"""Dimensionless tracking residual that remains informative far from target."""
import torch


def normalized_pseudo_huber(error, scale):
    if scale <= 0:
        raise ValueError('Expected positive metric scale')
    return torch.sqrt(1. + (error / scale).square().sum(-1)) - 1.


def foot_residual_cost(env, command_name, body_names, scale):
    c = env.command_manager.get_term(command_name)
    ix = [c.cfg.body_names.index(n) for n in body_names]
    error = c.body_pos_relative_w[:, ix] - c.robot_body_pos_w[:, ix]
    return normalized_pseudo_huber(error, scale).mean(-1)
