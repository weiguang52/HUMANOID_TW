#!/usr/bin/env python3
"""Read-only comparison of native and training FK before clipping.

Outputs evidence, not an acceptance gate: link frames can differ between models.
Endpoint correspondences must be reviewed before interpreting discrepancies.
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from retarget_humanml3d import (DEFAULT_ROOT, DEFAULT_TRAINING_URDF, MAPPING,
                               CUSTOM_JOINT_NAMES, map_28_to_30, load_limits)
from native_retarget import NATIVE_JOINT_NAMES


def fk(path, angles):
    root = ET.parse(path).getroot()
    joints = list(root.findall('joint'))
    children = {j.find('child').get('link') for j in joints}
    roots = {l.get('name') for l in root.findall('link')} - children
    if len(roots) != 1:
        raise ValueError('Expected one tree root')
    frames = {roots.pop(): np.eye(4)}
    while joints:
        remaining = []
        for j in joints:
            parent = j.find('parent').get('link')
            if parent not in frames:
                remaining.append(j)
                continue
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            transform[:3, 3] = xyz
            motion = np.eye(4)
            if j.get('type') in ('revolute', 'continuous'):
                axis_node = j.find('axis')
                axis = np.fromstring(axis_node.get('xyz'), sep=' ') if axis_node is not None else np.array([1., 0, 0])
                axis /= np.linalg.norm(axis)
                motion[:3, :3] = Rotation.from_rotvec(axis * angles.get(j.get('name'), 0.)).as_matrix()
            elif j.get('type') != 'fixed':
                raise ValueError('Unsupported joint type')
            frames[j.find('child').get('link')] = frames[parent] @ transform @ motion
        if len(remaining) == len(joints):
            raise ValueError('Disconnected joint tree')
        joints = remaining
    return frames


PAIRS = {'left_knee': ('left_calf', 'left_mid_leg'),
         'right_knee': ('right_calf', 'right_mid_leg'),
         'left_ankle': ('left_ankle', 'left_calf'),
         'right_ankle': ('right_ankle', 'right_calf'),
         'left_foot': ('left_foot', 'left_foot'),
         'right_foot': ('right_foot', 'right_foot'),
         'left_elbow': ('left_force_arm', 'left_force_arm'),
         'right_elbow': ('right_force_arm', 'right_force_arm'),
         'left_wrist': ('left_hand', 'left_wrist'),
         'right_wrist': ('right_hand', 'right_wrist'),
         'chest': ('chest', 'chest'), 'head': ('head', 'head')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    solver = DEFAULT_ROOT / 'assets/urdf/Assembly.urdf'
    source = (DEFAULT_ROOT / 'src/robot.cpp').read_text()
    block = re.search(r'limits\[28\]\[2\]\s*=\s*\{(.*?)\};', source, re.S).group(1)
    bounds = np.array([[float(x.strip().rstrip('f')) for x in pair.split(',')]
                       for pair in re.findall(r'\{([^{}]+)\}', block)])
    assert bounds.shape == (28, 2)
    lower, upper, _ = load_limits(DEFAULT_TRAINING_URDF)
    differences = []
    for i, name in enumerate(CUSTOM_JOINT_NAMES):
        old, sign, offset = MAPPING[name]
        if old is None:
            continue
        native_range = sorted((bounds[NATIVE_JOINT_NAMES.index(old)] * sign + offset).tolist())
        if not np.allclose(native_range, [lower[i], upper[i]], atol=.002):
            differences.append(dict(joint=name, native_mapped_deg=np.rad2deg(native_range).tolist(),
                                    training_deg=np.rad2deg([lower[i], upper[i]]).tolist()))
    neutral = np.zeros(28)
    neutral[16], neutral[21] = -np.pi/2, np.pi/2
    samples = [neutral]
    for i in range(28):
        for delta in [-.1, .1]:
            sample = neutral.copy()
            sample[i] += delta
            samples.append(sample)
    errors = {k: [] for k in PAIRS}
    neutral_positions = {}
    for index, q in enumerate(samples):
        a = fk(solver, dict(zip(NATIVE_JOINT_NAMES, q)))
        mapped = map_28_to_30(q[None], NATIVE_JOINT_NAMES)[0]
        b = fk(DEFAULT_TRAINING_URDF, dict(zip(CUSTOM_JOINT_NAMES, mapped)))
        for name, (native, training) in PAIRS.items():
            pa, pb = a[native][:3, 3], b[training][:3, 3]
            errors[name].append(float(np.linalg.norm(pa-pb)))
            if index == 0:
                neutral_positions[name] = {'native_m': pa.tolist(), 'training_m': pb.tolist()}
    report = {'note': 'Unclipped FK; base frames aligned, paired joint origins, no world root trajectory. Diagnostic poses need not be within limits.',
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [solver, DEFAULT_TRAINING_URDF]},
              'limit_differences': differences, 'neutral_positions': neutral_positions,
              'position_error_m': {k: {'neutral': v[0], 'max_57_probes': max(v)} for k,v in errors.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['limit_differences', 'position_error_m']}, indent=2))


if __name__ == '__main__':
    main()
