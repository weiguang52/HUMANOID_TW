"""Reference-preserving knee residual; original contact mask, no phase edits."""
import math
import torch
from .expression import reference_contacts


def masked_knee_cost(error, contact, known, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')
    # Average over both feet, not just known frames: no confidence amplification.
    mask = known.bool() & ~contact.bool()
    return ((torch.sqrt(1 + (error / scale).square()) - 1) * mask).mean(-1)


def swing_knee_cost(env, command_name, scale=.35):
    c = env.command_manager.get_term(command_name)
    names = ['left_knee_pitch_joint', 'right_knee_pitch_joint']
    ix = [c.cfg.joint_names.index(n) for n in names]
    contact, known = reference_contacts(c)
    return masked_knee_cost(c.joint_pos[:, ix] - c.robot_joint_pos[:, ix],
                            contact, known, scale)
