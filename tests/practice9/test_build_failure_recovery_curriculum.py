from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[2] / 'scripts/practice9/build_failure_recovery_curriculum.py'


def load_module():
    spec = importlib.util.spec_from_file_location('build_failure_recovery_curriculum', MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_motion(path: Path, length: int, support_start: int) -> None:
    body_names = np.asarray(['base_link', 'left_foot', 'right_foot'])
    positions = np.zeros((length, 3, 3), dtype=np.float32)
    positions[:, 0, 2] = 0.5
    positions[:, 1:, 2] = 0.3
    positions[support_start:, 1, 2] = 0.02
    velocities = np.zeros_like(positions)
    velocities[:support_start, 1:, 0] = 0.5
    np.savez(
        path,
        joint_pos=np.zeros((length, 2), dtype=np.float32),
        body_names=body_names,
        body_pos_w=positions,
        body_lin_vel_w=velocities,
    )


class FailureRecoveryCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_builds_trim_and_phase_prior_without_copying_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            early_path = root / 'early.npz'
            late_path = root / 'late.npz'
            write_motion(early_path, length=8, support_start=3)
            write_motion(late_path, length=10, support_start=0)
            source = {
                'schema_version': 1,
                'motions': [
                    {'id': 'early', 'file': str(early_path), 'frames': 8, 'quality_pass': True},
                    {'id': 'late', 'file': str(late_path), 'frames': 10, 'quality_pass': True},
                ],
            }
            source_path = root / 'source.json'
            source_path.write_text(json.dumps(source), encoding='utf-8')
            traces = {
                'schema_version': 1,
                'manifest_sha256': self.module.sha256(source_path),
                'records_captured': 4,
                'truncated': False,
                'records': [
                    {'motion_index': 0, 'motion_id': 'early', 'terminal': {'motion_frame': 1}},
                    {'motion_index': 0, 'motion_id': 'early', 'terminal': {'motion_frame': 1}},
                    {'motion_index': 1, 'motion_id': 'late', 'terminal': {'motion_frame': 5}},
                    {'motion_index': 1, 'motion_id': 'late', 'terminal': {'motion_frame': 6}},
                ],
            }
            traces_path = root / 'traces.json'
            traces_path.write_text(json.dumps(traces), encoding='utf-8')

            result = self.module.build_recovery_manifest(
                source,
                traces,
                source_path=source_path,
                traces_path=traces_path,
                support_height=0.05,
                support_speed=0.15,
                support_window_frames=2,
                early_failure_max_frame=2,
                phase_pre_frames=2,
                phase_post_frames=1,
                phase_multiplier=4.0,
            )

            early, late = result['motions']
            self.assertEqual(early['start_frame'], 3)
            self.assertEqual(early['frames'], 5)
            self.assertEqual(early['recovery']['mode'], 'trim_unstable_prefix')
            self.assertEqual(late['sampling_intervals'][0], {
                'start': 3,
                'end': 8,
                'multiplier': 4.0,
                'reason': 'repeatable_terminal_failure_phase',
            })
            self.assertEqual(late['recovery']['terminal_frames'], [5, 6])
            self.assertTrue(early_path.is_file())
            self.assertTrue(late_path.is_file())

    def test_rejects_missing_stable_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'motion.npz'
            write_motion(path, length=5, support_start=5)
            with self.assertRaisesRegex(ValueError, 'no stable support window'):
                self.module.find_stable_support_start(
                    path,
                    current_start=0,
                    current_end=None,
                    support_height=0.05,
                    support_speed=0.15,
                    support_window_frames=2,
                )


if __name__ == '__main__':
    unittest.main()
