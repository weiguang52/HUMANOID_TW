"""Integration checks against the server's actual CAD and IK models."""
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/practice9'))
from align_custom_robot_urdf import align, DEFAULT_REFERENCE, DEFAULT_OUTPUT
from joint_alignment import MAPPING
from retarget_humanml3d import map_28_to_30, CUSTOM_JOINT_NAMES


@unittest.skipUnless(DEFAULT_REFERENCE.is_file() and DEFAULT_OUTPUT.is_file(), 'Server URDF assets required')
class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'aligned.urdf'

    def test_real_models_preserve_fk_and_limits(self):
        report = align(DEFAULT_OUTPUT, DEFAULT_REFERENCE, self.output)
        self.assertEqual(report['mapped_joint_count'], 28)
        self.assertLess(report['max_fk_matrix_error'], 1e-10)
        before = {j.get('name'): j for j in ET.parse(DEFAULT_OUTPUT).getroot().findall('joint')}
        after = {j.get('name'): j for j in ET.parse(self.output).getroot().findall('joint')}
        self.assertEqual(sum(j.get('type') != 'fixed' for j in after.values()), 30)
        for new, (old, sign, offset) in MAPPING.items():
            if old is None:
                self.assertIn(new, after)
                continue
            self.assertNotIn(new, after)
            a = before[new].find('limit')
            b = after[old].find('limit')
            expected = sorted(sign * (float(a.get(k)) - offset) for k in ('lower', 'upper'))
            np.testing.assert_allclose([float(b.get('lower')), float(b.get('upper'))], expected)
            for key in ('effort', 'velocity'):
                self.assertEqual(a.get(key), b.get(key))

    def test_changed_axis_is_rejected(self):
        tree = ET.parse(DEFAULT_OUTPUT)
        tree.getroot().find("joint[@name='left_hip_linkage_pitch']/axis").set('xyz', '1 0 0')
        bad = Path(self.temp.name) / 'bad.urdf'
        tree.write(bad)
        with self.assertRaisesRegex(ValueError, 'Axis mismatch'):
            align(bad, DEFAULT_REFERENCE, self.output)
        self.assertFalse(self.output.exists())

    def test_inputs_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'differ'):
            align(DEFAULT_OUTPUT, DEFAULT_REFERENCE, DEFAULT_OUTPUT)

    def test_mapping_handles_permuted_motion_columns(self):
        names = [old for old, _, _ in MAPPING.values() if old is not None][::-1]
        old_q = np.arange(56, dtype=float).reshape(2, 28) / 100
        new_q = map_28_to_30(old_q, names)
        for column, new_name in enumerate(CUSTOM_JOINT_NAMES):
            old, sign, offset = MAPPING[new_name]
            expected = np.zeros(2) if old is None else sign * old_q[:, names.index(old)] + offset
            np.testing.assert_allclose(new_q[:, column], expected)


if __name__ == '__main__':
    unittest.main()
