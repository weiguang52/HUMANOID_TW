"""Reward-only stationary-support ablation; no changes to human motion timing.

Method inspiration: BeyondMimic global velocity tracking and ProtoMotions
reference contact matching. No upstream code copied.
"""
import math
import torch


def support_velocity_cost(reference_xy, actual_xy, reference_yaw_rate,
                          standing, contact, known, scale=0.02,
                          speed_low=0.005, speed_high=0.025,
                          yaw_low=0.1, yaw_high=0.3):
    """Bounded [N] cost on audited standing with two known support labels.

    Continuous gates exclude moving/turning references. Track small intentional
    reference velocity instead of zero. Reference contact gates prevent an agent
    evading this penalty by losing actual support. Units: m/s and rad/s.
    """
    if not all(math.isfinite(v) for v in (scale, speed_low, speed_high, yaw_low, yaw_high)):
        raise ValueError('Finite scales and thresholds required')
    if scale <= 0 or not 0 <= speed_low < speed_high or not 0 <= yaw_low < yaw_high:
        raise ValueError('Positive scale and ordered nonnegative thresholds required')
    speed = torch.linalg.vector_norm(reference_xy, dim=-1)
    speed_gate = ((speed_high - speed) / (speed_high - speed_low)).clamp(0, 1)
    yaw_gate = ((yaw_high - reference_yaw_rate.abs()) / (yaw_high - yaw_low)).clamp(0, 1)
    eligible = standing.bool() & contact.bool().all(-1) & known.bool().all(-1)
    error = ((actual_xy - reference_xy) / scale).square().sum(-1)
    return -torch.expm1(-error) * speed_gate * yaw_gate * eligible


def stationary_support_velocity(env, command_name, scale=0.02):
    from .expression import reference_contacts
    from .stability_tracking import standing_mask
    command = env.command_manager.get_term(command_name)
    contact, known = reference_contacts(command)
    index = command.cfg.body_names.index('base_link')
    return support_velocity_cost(
        command.body_lin_vel_w[:, index, :2],
        command.robot_body_lin_vel_w[:, index, :2],
        command.body_ang_vel_w[:, index, 2],
        standing_mask(command), contact, known, scale=scale,
    )
