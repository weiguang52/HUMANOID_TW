"""TW140: clip-safe leg previews and original-phase swing-foot tracking."""
import torch
from .expression import reference_contacts


def preview_indices(command, offsets):
    lengths = command.motion.clip_lengths[command.motion_ids]
    return [command.motion.frame_indices(command.motion_ids,
        torch.minimum(command.time_steps + offset, lengths - 1)) for offset in offsets]


def leg_preview(env, command_name, joint_names, horizons=(0., .1, .3)):
    c = env.command_manager.get_term(command_name)
    reference_contacts(c)
    labels, known = c._contact_reference
    ix = [c.cfg.joint_names.index(n) for n in joint_names]
    offsets = [round(t / env.step_dt) for t in horizons]
    values = []
    for frames in preview_indices(c, offsets):
        values.extend((c.motion.joint_pos[frames][:, ix],
                       labels[frames].float(), known[frames].float()))
    return torch.cat(values, dim=-1)


def masked_swing_score(error_z, contact, known, std):
    mask = known & ~contact
    score = torch.exp(-error_z.square() / std**2)
    return (score * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def swing_height_tracking(env, command_name, body_names, std):
    c = env.command_manager.get_term(command_name)
    ix = [c.cfg.body_names.index(n) for n in body_names]
    labels, known = reference_contacts(c)
    error = c.body_pos_relative_w[:, ix, 2] - c.robot_body_pos_w[:, ix, 2]
    return masked_swing_score(error, labels, known, std)
