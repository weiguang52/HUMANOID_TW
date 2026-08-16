from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[2] / "scripts/practice9/evaluation_utils.py"

spec = importlib.util.spec_from_file_location("practice9_evaluation_utils", MODULE)
assert spec and spec.loader
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)


class EvaluationUtilsTests(unittest.TestCase):
    def test_failure_trace_buffer_keeps_tail_and_ignores_success(self) -> None:
        buffer = utils.FailureTraceBuffer(environment_count=2, history_frames=2, max_records=1)
        buffer.append([{'frame': 1}, {'frame': 10}])
        buffer.append([{'frame': 2}, {'frame': 11}])
        buffer.append([{'frame': 3}, {'frame': 12}])
        buffer.complete(
            env_index=0,
            motion_index=7,
            motion_id='failed',
            episode=1,
            successful=False,
            terminal_snapshot={'frame': 4},
            termination_causes=['ee_body_pos'],
        )
        buffer.complete(
            env_index=1,
            motion_index=8,
            motion_id='success',
            episode=1,
            successful=True,
            terminal_snapshot=None,
            termination_causes=['motion_end'],
        )
        self.assertEqual(buffer.history_frames, 2)
        self.assertEqual(len(buffer.records), 1)
        self.assertEqual(buffer.records[0]['history'], [{'frame': 2}, {'frame': 3}])
        self.assertEqual(buffer.records[0]['terminal'], {'frame': 4})

    def test_atomic_write_reports_can_publish_failure_traces(self) -> None:
        payload = {'summary': {'motions': 1, 'episodes': 1, 'success_rate': 0.0, 'mean_completion_fraction': 0.5}}
        row = {'motion_index': 0, 'motion_id': 'clip', 'success_rate': 0.0, 'mean_completion_fraction': 0.5}
        traces = {'schema_version': 1, 'records': [{'motion_id': 'clip'}]}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'evaluation'
            utils.atomic_write_reports(output, payload, [row], traces)
            self.assertEqual(json.loads((output / 'failure_traces.json').read_text()), traces)
            self.assertFalse(any(output.glob('*.tmp')))

    def test_parse_motion_indices_defaults_to_all(self) -> None:
        self.assertEqual(utils.parse_motion_indices(None, 4), [0, 1, 2, 3])

    def test_parse_motion_indices_preserves_ranges_and_order(self) -> None:
        self.assertEqual(utils.parse_motion_indices("3,0-2,5", 6), [3, 0, 1, 2, 5])

    def test_parse_motion_indices_rejects_duplicate_and_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            utils.parse_motion_indices("0-2,2", 4)
        with self.assertRaisesRegex(ValueError, "outside"):
            utils.parse_motion_indices("4", 4)

    def test_build_report_uses_episode_and_step_denominators(self) -> None:
        metric_sums = {name: [4.0, 12.0] for name in utils.ERROR_METRICS}
        termination_counts = {name: [0, 0] for name in utils.TERMINATION_TERMS}
        termination_counts["motion_end"] = [2, 1]
        termination_counts["anchor_pos"] = [0, 1]
        summary, rows = utils.build_report(
            selected_indices=[2, 5],
            clip_ids=["a", "b"],
            clip_lengths=[100, 200],
            fps=50.0,
            episodes=[2, 2],
            successes=[2, 1],
            completion_sums=[2.0, 1.5],
            reward_sums=[20.0, 30.0],
            metric_steps=[10, 20],
            metric_sums=metric_sums,
            termination_counts=termination_counts,
        )
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["successes"], 3)
        self.assertAlmostEqual(summary["success_rate"], 0.75)
        self.assertEqual(summary["motions_full_success"], 1)
        self.assertEqual(summary["motions_zero_success"], 0)
        self.assertAlmostEqual(rows[0]["mean_error_anchor_pos"], 0.4)
        self.assertAlmostEqual(rows[1]["mean_error_anchor_pos"], 0.6)
        self.assertEqual(rows[1]["termination_anchor_pos"], 1)

    def test_atomic_write_reports_publishes_three_files(self) -> None:
        payload = {"summary": {"motions": 1, "episodes": 1, "success_rate": 1.0, "mean_completion_fraction": 1.0}}
        row = {
            "motion_index": 0,
            "motion_id": "clip",
            "success_rate": 1.0,
            "mean_completion_fraction": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "evaluation"
            utils.atomic_write_reports(output, payload, [row])
            self.assertTrue((output / "results.json").is_file())
            self.assertTrue((output / "per_motion.csv").is_file())
            self.assertTrue((output / "summary.txt").is_file())
            self.assertEqual(json.loads((output / "results.json").read_text())["summary"]["episodes"], 1)
            self.assertFalse(any(output.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
