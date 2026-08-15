from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / 'scripts/practice9/build_locomotion_curriculum.py'
SPEC = importlib.util.spec_from_file_location('curriculum_builder', SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def motion(motion_id: str, root_speed: float = 0.1) -> dict:
    return {
        'id': motion_id,
        'file': f'/data/{motion_id}.npz',
        'frames': 100,
        'quality_pass': True,
        'fk_quality_pass': True,
        'body_linear_velocity_error_p95': 0.01,
        'body_angular_velocity_error_p95': 0.01,
        'retarget_quality': {
            'velocity_ratio_p99_by_joint': {'joint': 0.2},
            'near_limit_fraction_by_joint': {'joint': 0.0},
            'root_xy_speed_p99': root_speed,
            'root_yaw_rate_p99': 0.1,
        },
    }


class CurriculumBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = {'motions': [motion('seed_a'), motion('seed_b')]}
        self.full = {
            'motions': [
                motion('forward_easy', 0.1),
                motion('forward_hard', 0.5),
                motion('turning', 0.2),
                motion('backward', 0.2),
                motion('sideways', 0.2),
            ]
        }
        tags = {
            'forward_easy': 'forward',
            'forward_hard': 'forward',
            'turning': 'turning',
            'backward': 'backward',
            'sideways': 'sideways',
        }
        self.selection = {'motions': [{'motion_id': key, 'tag': value} for key, value in tags.items()]}

    def test_stages_are_nested_and_keep_seeds(self) -> None:
        stages = builder.build_curriculum(self.full, self.seed, self.selection, [3, 5, 7])
        ids = [{record['id'] for record in stage['motions']} for stage in stages]
        self.assertEqual([len(value) for value in ids], [3, 5, 7])
        self.assertTrue({'seed_a', 'seed_b'} <= ids[0])
        self.assertTrue(ids[0] < ids[1] < ids[2])
        self.assertEqual(ids[-1], {record['id'] for record in self.seed['motions'] + self.full['motions']})

    def test_easier_motion_is_selected_first_within_category(self) -> None:
        stage = builder.build_curriculum(self.full, self.seed, self.selection, [3])[0]
        ids = {record['id'] for record in stage['motions']}
        self.assertIn('forward_easy', ids)
        self.assertNotIn('forward_hard', ids)

    def test_overlapping_seed_is_rejected(self) -> None:
        seed = {'motions': [motion('forward_easy')]}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            builder.build_curriculum(self.full, seed, self.selection, [2])


if __name__ == '__main__':
    unittest.main()
