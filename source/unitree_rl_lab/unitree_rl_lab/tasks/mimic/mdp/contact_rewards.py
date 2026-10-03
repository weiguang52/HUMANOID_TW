"""Opt-in phase supervision. Does not hide raw impact/dropout signals with debounce."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def phase_cost(force_z, contact, known, force_on=1.0):
    """force_z [N,H,2], labels [N,2]. Penalize every measured sample, not max."""
    if not np.isfinite(force_on) or force_on <= 0:
        raise ValueError('force_on must be positive')
    support = (force_z / force_on).clamp(0., 1.)
    target = contact[:, None, :].to(support.dtype)
    error = (support - target).square().mean(dim=1)
    mask = known.to(error.dtype)
    return (error * mask).sum(-1) / mask.sum(-1).clamp_min(1.)


def load_reference(command):
    manifest = Path(command.cfg.motion_file)
    data = json.loads(manifest.read_text())
    spec = data.get('contact_reference')
    if not spec:
        raise ValueError('Enabled contact reward requires contact_reference sidecar')
    path = Path(spec['file'])
    if not path.is_absolute():
        path = manifest.parent / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Contact sidecar hash mismatch')
    with np.load(path, allow_pickle=False) as a:
        ids = a['clip_ids'].tolist()
        lengths = a['lengths']
        if ids != command.motion.clip_ids:
            raise ValueError('Contact clip order mismatch')
        if not np.array_equal(lengths, command.motion.clip_lengths.cpu().numpy()):
            raise ValueError('Contact clip lengths/crops mismatch')
        if a['foot_names'].tolist() != ['left_foot', 'right_foot']:
            raise ValueError('Contact foot order mismatch')
        shape = (command.motion.time_step_total, 2)
        values = []
        for field in ['contact','known']:
            x = a[field]
            if x.shape != shape or not np.isin(x,[0,1]).all():
                raise ValueError('Invalid binary contact supervision')
            values.append(torch.as_tensor(x.copy(),device=command.device,dtype=torch.bool))
    return tuple(values)


def motion_contact_phase_cost(env, command_name, sensor_cfg, force_on=1.0):
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, '_contact_reference'):
        command._contact_reference = load_reference(command)
    labels, known = command._contact_reference
    frames = command.frame_indices
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    return phase_cost(forces, labels[frames], known[frames], force_on)
