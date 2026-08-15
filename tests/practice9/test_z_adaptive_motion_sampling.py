from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

# Kept last in unittest discovery because the HumanML3D CPU-guard tests assert
# that torch has not been imported by an earlier test module.

MODULE = (
    Path(__file__).parents[2]
    / 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/adaptive_sampling.py'
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


if __name__ == '__main__':
    unittest.main()
