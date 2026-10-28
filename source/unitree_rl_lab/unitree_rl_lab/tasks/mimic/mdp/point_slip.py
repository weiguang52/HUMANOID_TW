"""Ground-contact point speed, retaining per-environment ownership."""
import math
import torch


def point_speed_cost(forces, points, counts, starts, com, linear, angular, capacity=64, minimum_load=1.):
    if not math.isfinite(minimum_load) or minimum_load<=0:raise ValueError("Positive finite minimum load required")
    counts, starts = counts.reshape(-1), starts.reshape(-1)
    if counts.numel() != com.shape[0]:
        raise ValueError('Expected one foot and one ground filter per environment')
    if (counts > capacity).any():
        raise RuntimeError('Contact capacity exceeded; refusing truncated reward')
    offsets = torch.arange(capacity, device=com.device)[None, :]
    ids = starts[:, None].long() + offsets
    valid = offsets < counts[:, None]
    ids = ids.clamp(0, points.shape[0]-1)
    f = forces.reshape(-1)[ids]
    valid = valid & (f > 1e-6)
    p = torch.where(valid[..., None], points[ids], com[:, None, :])
    velocity = linear[:, None, :] + torch.cross(angular[:, None, :].expand_as(p), p-com[:, None, :], dim=-1)
    squared = torch.where(valid, velocity[..., :2].square().sum(-1), 0.)
    rms = (squared.sum(-1)/valid.sum(-1).clamp_min(1)).sqrt()
    load = torch.where(valid, f, 0.).sum(-1)
    return torch.where(load > minimum_load, rms, 0.)


def ground_point_slip(env, minimum_load=1.):
    robot = env.scene['robot']
    if not hasattr(env, '_point_slip_body_ids'):
        env._point_slip_body_ids = [robot.body_names.index(n) for n in ('left_foot','right_foot')]
    result = torch.zeros(env.num_envs, device=env.device)
    for name, body in zip(('left_foot','right_foot'), env._point_slip_body_ids):
        sensor = env.scene['slip_'+name]
        force, point, _, _, count, start = sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        data = robot.data
        result += point_speed_cost(force, point, count, start,
            data.body_com_pos_w[:,body], data.body_lin_vel_w[:,body], data.body_ang_vel_w[:,body], minimum_load=minimum_load)
    return result


def moving_slip_gate(reference_speed):
    u=((reference_speed-.05)/.05).clamp(0.,1.)
    return u*u*(3.-2.*u)


def normalized_moving_point_slip(env, command_name='motion', speed_scale=.05):
    if speed_scale<=0:raise ValueError('Positive speed normalization required')
    command=env.command_manager.get_term(command_name)
    index=command.cfg.body_names.index('base_link')
    speed=torch.linalg.vector_norm(command.body_lin_vel_w[:,index,:2],dim=-1)
    # Mean across the two feet; zero cost for unloaded feet, original contact rewards retained.
    return moving_slip_gate(speed)*ground_point_slip(env)/(2.*speed_scale)
