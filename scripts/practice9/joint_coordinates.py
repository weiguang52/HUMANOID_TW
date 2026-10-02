"""Load the simulator-independent shared joint contract for standalone scripts."""
import importlib.util
from pathlib import Path

_path = Path(__file__).parents[2] / 'source/unitree_rl_lab/unitree_rl_lab/assets/robots/custom_joint_coordinates.py'
_spec = importlib.util.spec_from_file_location('custom_joint_coordinates', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
COORDINATES = _module.COORDINATES
rename = _module.rename
convert_motion = _module.convert_motion


def apply_coordinates(root):
    """Apply the table once to a freshly generated CAD-coordinate URDF."""
    import math
    import xml.etree.ElementTree as ET
    import numpy as np

    def rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)

    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    missing = set(COORDINATES) - set(joints)
    if missing:
        raise ValueError(f'Missing original CAD joints: {sorted(missing)}')
    for original, (name, sign, initial) in COORDINATES.items():
        joint = joints[original]
        axis_element = joint.find('axis')
        axis = np.array([float(x) for x in axis_element.get('xyz').split()])
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError(f'Invalid axis: {original}')
        if initial:
            origin = joint.find('origin')
            if origin is None:
                origin = ET.SubElement(joint, 'origin', xyz='0 0 0')
            r, p, y = [float(x) for x in origin.get('rpy', '0 0 0').split()]
            matrix = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            # original q = sign * (aligned q - initial).
            matrix = matrix @ rotation(axis, -sign * initial)
            pitch = math.atan2(-matrix[2, 0], math.hypot(matrix[0, 0], matrix[1, 0]))
            if abs(math.cos(pitch)) > 1e-10:
                roll = math.atan2(matrix[2, 1], matrix[2, 2])
                yaw = math.atan2(matrix[1, 0], matrix[0, 0])
            else:
                roll = math.atan2(-matrix[1, 2], matrix[1, 1])
                yaw = 0.0
            origin.set('rpy', ' '.join(f'{v:.17g}' for v in (roll, pitch, yaw)))
        axis_element.set('xyz', ' '.join(f'{sign * v:.17g}' for v in axis))
        joint.set('name', name)
        limit = joint.find('limit')
        lower, upper = sorted(sign * float(limit.get(key)) + initial for key in ('lower', 'upper'))
        limit.set('lower', f'{lower:.17g}')
        limit.set('upper', f'{upper:.17g}')
    names = [j.get('name') for j in root.findall('joint')]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate aligned joint names')
