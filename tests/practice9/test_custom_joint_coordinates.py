import sys
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / 'scripts/practice9'))
sys.path.insert(0, str(REPO / 'source/unitree_rl_lab'))
from joint_coordinates import COORDINATES, apply_coordinates, convert_motion, rename
from prepare_custom_robot_urdf import generate, DEFAULT_SOURCE, DEFAULT_MESH_ROOT, DEFAULT_OUTPUT


def fk(root, q):
    joints = list(root.findall('joint'))
    child_links = {j.find('child').get('link') for j in joints}
    roots = {link.get('name') for link in root.findall('link')} - child_links
    transforms = {roots.pop(): np.eye(4)}
    while joints:
        count = len(joints)
        for joint in list(joints):
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            origin = joint.find('origin')
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            matrix[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                axis /= np.linalg.norm(axis)
                matrix[:3, :3] = matrix[:3, :3] @ Rotation.from_rotvec(axis * q.get(joint.get('name'), 0)).as_matrix()
            transforms[joint.find('child').get('link')] = transforms[parent] @ matrix
            joints.remove(joint)
        if len(joints) == count:
            raise ValueError('Invalid link tree')
    return transforms


class CoordinateTests(unittest.TestCase):
    def test_explicit_table_initial_readings(self):
        self.assertEqual(len(COORDINATES), 20)
        self.assertEqual(sum(sign == -1 for _, sign, _ in COORDINATES.values()), 6)
        shifts = {new: offset for new, _, offset in COORDINATES.values() if offset}
        self.assertEqual(shifts, {'left_shoulder_pitch_joint': -math.pi / 2,
                                  'right_shoulder_roll_joint': math.pi / 2})
        self.assertEqual(rename('left_wrist_pitch'), 'left_wrist_pitch')
        self.assertEqual(rename('left_foot_roll'), 'left_foot_roll')

    def test_motion_coordinate_conversion_is_not_applied_twice(self):
        names = ['right_upper_arm_roll', 'left_shoulder_linkage_pitch', 'left_wrist_pitch']
        pos = np.array([[0., 0., .2], [.1, -.3, -.4]])
        vel = np.ones_like(pos)
        targets = ['right_shoulder_roll_joint', 'left_shoulder_pitch_joint', 'left_wrist_pitch']
        aligned, converted, converted_vel = convert_motion(names, pos, vel, targets)
        self.assertEqual(aligned, targets)
        np.testing.assert_allclose(converted[:, 0], pos[:, 0] + math.pi / 2)
        np.testing.assert_allclose(converted[:, 1], -pos[:, 1] - math.pi / 2)
        np.testing.assert_allclose(converted_vel, [[1, -1, 1], [1, -1, 1]])
        _, twice, _ = convert_motion(aligned, converted, converted_vel, targets)
        np.testing.assert_array_equal(converted, twice)
        np.testing.assert_array_equal(pos, [[0, 0, .2], [.1, -.3, -.4]])

    def test_mixed_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            convert_motion(['left_thigh_roll', 'left_hip_roll_joint'], np.zeros((2, 2)),
                           np.zeros((2, 2)), ['left_hip_roll_joint'])

    @unittest.skipUnless(DEFAULT_SOURCE.is_file(), 'Server CAD assets required')
    def test_generator_is_deterministic_and_preserves_pose(self):
        backup = DEFAULT_OUTPUT.parent / 'backups/urdf0711_training_30dof.pre_tw44_table.urdf'
        self.assertTrue(backup.is_file())
        original = ET.parse(backup).getroot()
        original_joints = {j.get('name'): j for j in original.findall('joint')}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'training.urdf'
            meta = generate(DEFAULT_SOURCE, DEFAULT_MESH_ROOT, path)
            first = path.read_bytes()
            generate(DEFAULT_SOURCE, DEFAULT_MESH_ROOT, path)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(first, DEFAULT_OUTPUT.read_bytes())
            aligned = ET.parse(path).getroot()
            aligned_joints = {j.get('name'): j for j in aligned.findall('joint')}
            self.assertEqual(sum(j.get('type') != 'fixed' for j in aligned_joints.values()), 30)
            for old, (new, sign, offset) in COORDINATES.items():
                a, b = original_joints[old], aligned_joints[new]
                self.assertNotIn(old, aligned_joints)
                np.testing.assert_allclose(np.fromstring(b.find('axis').get('xyz'), sep=' '),
                                           sign * np.fromstring(a.find('axis').get('xyz'), sep=' '))
                limits = sorted(sign * float(a.find('limit').get(k)) + offset for k in ('lower', 'upper'))
                np.testing.assert_allclose([float(b.find('limit').get(k)) for k in ('lower', 'upper')], limits)
                for tag in ('parent', 'child'):
                    self.assertEqual(a.find(tag).attrib, b.find(tag).attrib)
            for link in original.findall('link'):
                other = aligned.find(f"link[@name='{link.get('name')}']")
                self.assertEqual(ET.tostring(link), ET.tostring(other))
            rng = np.random.default_rng(44)
            error = 0.0
            for sample in range(65):
                old_q = {name: (0.0 if sample == 0 else float(rng.uniform(float(j.find('limit').get('lower')), float(j.find('limit').get('upper')))))
                         for name, j in original_joints.items() if j.get('type') != 'fixed'}
                new_q = {rename(name): COORDINATES.get(name, (name, 1, 0.0))[1] * value
                         + COORDINATES.get(name, (name, 1, 0.0))[2] for name, value in old_q.items()}
                before, after = fk(original, old_q), fk(aligned, new_q)
                error = max(error, max(float(np.max(np.abs(before[name] - after[name]))) for name in before))
            self.assertLess(error, 1e-12)
            print(f'65 poses, all links, max FK error: {error:.3g}')
            self.assertEqual(meta['initial_joint_positions']['left_shoulder_pitch_joint'], -math.pi / 2)
            self.assertEqual(meta['initial_joint_positions']['right_shoulder_roll_joint'], math.pi / 2)


if __name__ == '__main__':
    unittest.main()
