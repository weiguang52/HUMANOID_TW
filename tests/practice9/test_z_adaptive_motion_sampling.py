from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Kept last in unittest discovery because the HumanML3D CPU-guard tests assert
# that torch has not been imported by an earlier test module.

MODULE = (
    Path(__file__).parents[2]
    / 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/adaptive_sampling.py'
)
MOTION_LIBRARY_MODULE = (
    Path(__file__).parents[2]
    / 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/motion_library.py'
)


class AdaptiveSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        spec = importlib.util.spec_from_file_location('adaptive_sampling', MODULE)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.torch = torch
        cls.sampling = module
        motion_spec = importlib.util.spec_from_file_location('practice9_motion_library', MOTION_LIBRARY_MODULE)
        assert motion_spec and motion_spec.loader
        motion_module = importlib.util.module_from_spec(motion_spec)
        sys.modules[motion_spec.name] = motion_module
        motion_spec.loader.exec_module(motion_module)
        cls.motion_library = motion_module

    def test_failure_rate_is_visit_normalized(self) -> None:
        failed = self.torch.tensor([1.0, 10.0, 0.0, 0.0])
        visited = self.torch.tensor([2.0, 100.0, 4.0, 0.0])
        actual = self.sampling.failure_rates(failed, visited, unvisited_score=1.0)
        self.torch.testing.assert_close(actual, self.torch.tensor([0.5, 0.1, 0.0, 1.0]))

    def test_uniform_ratio_is_exact_mixture_mass(self) -> None:
        score = self.torch.tensor([1.0, 0.0, 0.0, 0.0])
        prior = self.torch.ones(4)
        actual = self.sampling.balanced_sampling_probabilities(
            score,
            prior,
            uniform_ratio=0.75,
        )
        self.torch.testing.assert_close(actual, self.torch.tensor([0.4375, 0.1875, 0.1875, 0.1875]))

    def test_zero_difficulty_uses_prior(self) -> None:
        actual = self.sampling.balanced_sampling_probabilities(
            self.torch.zeros(3),
            self.torch.tensor([1.0, 2.0, 1.0]),
            uniform_ratio=0.7,
        )
        self.torch.testing.assert_close(actual, self.torch.tensor([0.25, 0.5, 0.25]))

    def test_probability_cap_redistributes_mass(self) -> None:
        actual = self.sampling.balanced_sampling_probabilities(
            self.torch.tensor([1.0, 0.0, 0.0, 0.0]),
            self.torch.ones(4),
            uniform_ratio=0.0,
            max_probability=0.4,
        )
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=6)
        self.assertLessEqual(float(actual.max()), 0.4 + 1.0e-6)
        self.torch.testing.assert_close(actual[1:], self.torch.full((3,), 0.2))

    def test_infeasible_cap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            self.sampling.balanced_sampling_probabilities(
                self.torch.ones(4),
                self.torch.ones(4),
                uniform_ratio=0.7,
                max_probability=0.2,
            )

    def test_motion_library_applies_frame_slice_and_phase_prior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            motion_path = root / 'motion.npz'
            length = 6
            body_quat = np.zeros((length, 1, 4), dtype=np.float32)
            body_quat[:, :, 0] = 1.0
            np.savez(
                motion_path,
                fps=np.asarray(50.0, dtype=np.float32),
                joint_names=np.asarray(['joint']),
                body_names=np.asarray(['base']),
                joint_pos=np.arange(length, dtype=np.float32).reshape(length, 1),
                joint_vel=np.zeros((length, 1), dtype=np.float32),
                body_pos_w=np.zeros((length, 1, 3), dtype=np.float32),
                body_quat_w=body_quat,
                body_lin_vel_w=np.zeros((length, 1, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((length, 1, 3), dtype=np.float32),
            )
            manifest = {
                'schema_version': 1,
                'motions': [
                    {
                        'id': 'motion',
                        'file': str(motion_path),
                        'weight': 2.0,
                        'quality_pass': True,
                        'start_frame': 1,
                        'end_frame': 5,
                        'sampling_intervals': [{'start': 1, 'end': 3, 'multiplier': 3.0}],
                    }
                ],
            }
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            library = self.motion_library.MotionLibrary(
                manifest_path,
                device='cpu',
                tracked_body_names=['base'],
                legacy_body_indexes=[0],
                joint_names=['joint'],
                expected_fps=50.0,
                max_frames=100,
                require_quality_pass=True,
            )
            self.assertEqual(library.clip_lengths.tolist(), [4])
            self.torch.testing.assert_close(
                library.joint_pos[:, 0],
                self.torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
            self.assertEqual(library.build_bins(0.04), 2)
            self.torch.testing.assert_close(library.bin_weights, self.torch.tensor([8.0, 6.0]))


if __name__ == '__main__':
    unittest.main()
