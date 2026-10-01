#!/usr/bin/env python3
"""Reparameterize new URDF with old joint names and positive rotation conventions.

q_new = sign * q_old + offset. Geometry is preserved exactly. This does not
make the different robot dimensions or waist joint chain orders identical.
"""
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from joint_alignment import MAPPING
from prepare_custom_robot_urdf import DEFAULT_OUTPUT

DEFAULT_REFERENCE = Path('/root/gpufree-data/projects/IK_V2/data/urdf/Assembly.urdf')


def vector(element, attr, default):
    values = np.array([float(x) for x in element.get(attr, default).split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f'Invalid {attr}: {values}')
    return values


def rotation(axis, angle):
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid rotation axis')
    x, y, z = np.asarray(axis) / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def rpy_matrix(rpy):
    r, p, y = rpy
    return rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)


def matrix_rpy(matrix):
    p = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(math.cos(p)) > 1e-10:
        return [math.atan2(matrix[2, 1], matrix[2, 2]), p, math.atan2(matrix[1, 0], matrix[0, 0])]
    return [math.atan2(-matrix[1, 2], matrix[1, 1]), p, 0.0]


def format_vector(values):
    return ' '.join(f'{value:.17g}' for value in values)


def forward_kinematics(root, positions):
    joints = list(root.findall('joint'))
    links = {link.get('name') for link in root.findall('link')}
    children = {joint.find('child').get('link') for joint in joints}
    roots = links - children
    if len(roots) != 1:
        raise ValueError('Expected one root link')
    transforms = {roots.pop(): np.eye(4)}
    axes = {}
    while joints:
        progress = False
        for joint in list(joints):
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            origin = joint.find('origin')
            if origin is None:
                origin = ET.Element('origin')
            local = np.eye(4)
            local[:3, :3] = rpy_matrix(vector(origin, 'rpy', '0 0 0'))
            local[:3, 3] = vector(origin, 'xyz', '0 0 0')
            world = transforms[parent] @ local
            if joint.get('type') != 'fixed':
                if joint.get('type') not in ('revolute', 'continuous'):
                    raise ValueError('Only rotational and fixed joints supported')
                element = joint.find('axis')
                if element is None:
                    element = ET.Element('axis')
                axis = vector(element, 'xyz', '1 0 0')
                axis = axis / np.linalg.norm(axis)
                if not np.isfinite(axis).all():
                    raise ValueError('Invalid axis')
                axes[joint.get('name')] = world[:3, :3] @ axis
                world[:3, :3] = world[:3, :3] @ rotation(axis, positions.get(joint.get('name'), 0))
            child = joint.find('child').get('link')
            if child in transforms:
                raise ValueError('Duplicate child link')
            transforms[child] = world
            joints.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Disconnected or cyclic URDF')
    return transforms, axes


def align(source, reference, output):
    if output.resolve() in (source.resolve(), reference.resolve()):
        raise ValueError('Output must differ from inputs')
    original = ET.parse(source).getroot()
    old = ET.parse(reference).getroot()
    root = ET.fromstring(ET.tostring(original))
    source_joints = {j.get('name'): j for j in root.findall('joint')}
    if len(source_joints) != len(root.findall('joint')):
        raise ValueError('Duplicate source joint names')
    offsets = {name: offset for name, (_, _, offset) in MAPPING.items()}
    _, old_axes = forward_kinematics(old, {})
    _, new_axes = forward_kinematics(original, offsets)
    rows = []
    for new_name, (old_name, sign, offset) in MAPPING.items():
        joint = source_joints[new_name]
        if old_name is None:
            continue
        if joint.find('mimic') is not None:
            raise ValueError('Mimic joints need separate conversion')
        dot = float(old_axes[old_name] @ (sign * new_axes[new_name]))
        if not math.isclose(dot, 1, abs_tol=1e-8):
            raise ValueError(f'Axis mismatch for {new_name}: {dot}')
        origin = joint.find('origin')
        if origin is None:
            origin = ET.SubElement(joint, 'origin', xyz='0 0 0')
        axis = vector(joint.find('axis'), 'xyz', '1 0 0')
        rotated = rpy_matrix(vector(origin, 'rpy', '0 0 0')) @ rotation(axis, offset)
        origin.set('rpy', format_vector(matrix_rpy(rotated)))
        joint.find('axis').set('xyz', format_vector(sign * axis))
        joint.set('name', old_name)
        limit = joint.find('limit')
        if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
            bounds = sorted(sign * (float(limit.get(key)) - offset) for key in ('lower', 'upper'))
            limit.set('lower', f'{bounds[0]:.17g}')
            limit.set('upper', f'{bounds[1]:.17g}')
        rows.append(dict(new_name=new_name, old_name=old_name, sign=sign,
                         offset_rad=offset, aligned_axis_dot=dot))
    names = [joint.get('name') for joint in root.findall('joint')]
    if len(names) != len(set(names)):
        raise ValueError('Alignment produces duplicate names')
    rng = np.random.default_rng(44)
    max_error = 0.0
    for sample in range(33):
        q_old = {name: (0.0 if sample == 0 else float(rng.uniform(-0.4, 0.4))) for name in names}
        q_new = dict(q_old)
        for new_name, (old_name, sign, offset) in MAPPING.items():
            if old_name is not None:
                q_new[new_name] = sign * q_old[old_name] + offset
        before, _ = forward_kinematics(original, q_new)
        after, _ = forward_kinematics(root, q_old)
        max_error = max(max_error, max(float(np.max(np.abs(before[n] - after[n]))) for n in before))
    if max_error > 1e-10:
        raise ValueError(f'FK reparameterization failed: {max_error}')
    root.set('name', original.get('name', 'custom_robot') + '_old_joint_coordinates')
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    report = dict(schema_version=1, source=str(source.resolve()), reference=str(reference.resolve()),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  output=str(output.resolve()), joints=rows, mapped_joint_count=len(rows),
                  extra_active_joints=[j.get('name') for j in root.findall('joint')
                                       if j.get('type') != 'fixed'
                                       and j.get('name') not in {old for old, _, _ in MAPPING.values()}],
                  fk_pose_count=33, max_fk_matrix_error=max_error,
                  limitations=['Robot dimensions and waist chain order differ.',
                               'Motion files need FK regeneration and matching RL configuration.'])
    output.with_suffix('.alignment.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT.with_name('urdf0711_old_joint_coordinates_30dof.urdf'))
    args = parser.parse_args()
    print(json.dumps(align(args.source, args.reference, args.output), indent=2))


if __name__ == '__main__':
    main()
