"""TW202 opt-in reward groups. Never change human contact or motion phase."""
import json
from pathlib import Path
import torch


def huber(error, scale):
    if scale <= 0:
        raise ValueError('Positive scale required')
    return torch.sqrt(1 + (error / scale).square().sum(-1)) - 1


def standing_mask(c):
    if not hasattr(c, '_tw202_standing'):
        rows = json.loads(Path(c.cfg.motion_file).read_text())['motions']
        if [r['id'] for r in rows] != c.motion.clip_ids:
            raise ValueError('Motion category ordering mismatch')
        c._tw202_standing = torch.tensor([r.get('category') == 'standing_upper' for r in rows], device=c.device)
    return c._tw202_standing[c.motion_ids]


def precise_path_cost(env, command_name, walking_scale, standing_scale, speed_threshold=.025):
    c = env.command_manager.get_term(command_name)
    stationary = standing_mask(c) & (c.anchor_lin_vel_w[:, :2].norm(dim=-1) < speed_threshold)
    error = c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2]
    return torch.where(stationary, huber(error, standing_scale), huber(error, walking_scale))


def neck_tracking_cost(env, command_name, scale, velocity=False):
    c = env.command_manager.get_term(command_name)
    indexes = [c.cfg.joint_names.index(n) for n in ('neck', 'neck_linkage_roll', 'head_pitch')]
    reference = c.joint_vel if velocity else c.joint_pos
    actual = c.robot_joint_vel if velocity else c.robot_joint_pos
    # Mean across the three axes keeps group size from multiplying the penalty.
    error = (reference[:, indexes] - actual[:, indexes]) / scale
    return (torch.sqrt(1 + error.square()) - 1).mean(-1)


def yaw_error(reference, actual):
    def yaw(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    delta = yaw(reference) - yaw(actual)
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def pelvis_heading_cost(env, command_name, scale):
    c = env.command_manager.get_term(command_name)
    i = c.cfg.body_names.index('base_link')
    error = yaw_error(c.body_quat_w[:, i], c.robot_body_quat_w[:, i])
    return huber(error.unsqueeze(-1), scale)
