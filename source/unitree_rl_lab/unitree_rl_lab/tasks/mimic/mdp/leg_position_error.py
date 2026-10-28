"""Opt-in reference-speed-gated leg joint position cost."""
import math
import torch


def leg_position_cost(actual, reference, reference_speed, scale=0.5):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Position scale must be finite and positive')
    if actual.ndim != 2 or actual.shape != reference.shape:
        raise ValueError('Expected matching environment-by-joint tensors')
    if reference_speed.shape != actual.shape[:1]:
        raise ValueError('Expected one reference speed per environment')
    u = ((reference_speed - .05) / .05).clamp(0., 1.)
    gate = u * u * (3. - 2. * u)
    return gate * ((actual - reference) / scale).square().mean(-1)


def moving_leg_position_error(env, command_name='motion', scale=0.5):
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, '_position_cost_leg_ids'):
        names = [command.robot.joint_names[i] for i in command.joint_indexes.tolist()]
        expected = [side + '_' + joint + '_joint' for side in ('left', 'right')
                    for joint in ('hip_pitch', 'hip_roll', 'hip_yaw',
                                  'knee_pitch', 'ankle_yaw', 'ankle_pitch')]
        if any(names.count(n) != 1 for n in expected):
            raise ValueError('Expected all twelve leg joints exactly once')
        command._position_cost_leg_ids = [names.index(n) for n in expected]
    ids = command._position_cost_leg_ids
    root = command.cfg.body_names.index('base_link')
    speed = torch.linalg.vector_norm(command.body_lin_vel_w[:, root, :2], dim=-1)
    return leg_position_cost(command.robot_joint_pos[:, ids], command.joint_pos[:, ids], speed, scale)
