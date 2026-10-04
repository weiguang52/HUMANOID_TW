"""TW100 expression goals with immutable, frame-aligned human contact phases."""
import torch
from .contact_rewards import load_reference


def reference_contacts(command):
    if not hasattr(command, '_contact_reference'):
        command._contact_reference = load_reference(command)
    labels, known = command._contact_reference
    return labels[command.frame_indices], known[command.frame_indices]


def upper_command(env, command_name, joint_names):
    command = env.command_manager.get_term(command_name)
    indices = [command.cfg.joint_names.index(n) for n in joint_names]
    return torch.cat((command.joint_pos[:, indices], command.joint_vel[:, indices]), -1)


def contact_command(env, command_name):
    """Expose current labels AND confidence, never treat unknown as swing."""
    command = env.command_manager.get_term(command_name)
    labels, known = reference_contacts(command)
    return torch.cat((labels.float(), known.float()), -1)


def anchor_height_error(env, command_name):
    c = env.command_manager.get_term(command_name)
    return (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).unsqueeze(-1)


def height_tracking(env, command_name, std):
    return torch.exp(-anchor_height_error(env, command_name).squeeze(-1).square() / std**2)


def upper_velocity_cost(env, command_name, joint_names):
    c = env.command_manager.get_term(command_name)
    ix = [c.cfg.joint_names.index(n) for n in joint_names]
    return (c.robot_joint_vel[:, ix] - c.joint_vel[:, ix]).square().mean(-1)


def standing_drift(env, command_name, std, speed_threshold=0.025):
    """Use audited motion CATEGORY, not root speed alone (treadmill != stand)."""
    c = env.command_manager.get_term(command_name)
    if not hasattr(c, '_expression_standing'):
        import json
        from pathlib import Path
        rows = json.loads(Path(c.cfg.motion_file).read_text())['motions']
        if [r['id'] for r in rows] != c.motion.clip_ids:
            raise ValueError('Standing category order mismatch')
        c._expression_standing = torch.tensor(
            [r.get('category') == 'standing_upper' for r in rows], device=c.device)
    standing = c._expression_standing[c.motion_ids]
    stationary = c.anchor_lin_vel_w[:, :2].norm(dim=-1) < speed_threshold
    error = (c.anchor_pos_w[:, :2] - c.robot_anchor_pos_w[:, :2]).square().sum(-1)
    return (1.0 - torch.exp(-error / std**2)) * standing * stationary
