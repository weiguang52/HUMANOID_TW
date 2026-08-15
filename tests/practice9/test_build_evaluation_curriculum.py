from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts/practice9/build_evaluation_curriculum.py"
SPEC = importlib.util.spec_from_file_location("evaluation_curriculum", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class EvaluationCurriculumTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, dict, dict]:
        source = {
            "schema_version": 1,
            "target_fps": 50.0,
            "motions": [
                {"id": "full", "file": "/data/full.npz", "frames": 100, "weight": 1.0},
                {"id": "partial", "file": "/data/partial.npz", "frames": 200, "weight": 1.0},
                {"id": "zero", "file": "/data/zero.npz", "frames": 300, "weight": 1.0},
            ],
        }
        source_path = root / "source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        evaluation = {
            "schema_version": 1,
            "manifest_sha256": builder.sha256(source_path),
            "checkpoint": "/data/model.pt",
            "checkpoint_sha256": "a" * 64,
            "episodes_per_motion": 3,
            "selected_motion_indices": [0, 1, 2],
            "motions": [
                {"motion_index": 0, "motion_id": "full", "episodes": 3, "successes": 3},
                {"motion_index": 1, "motion_id": "partial", "episodes": 3, "successes": 2},
                {"motion_index": 2, "motion_id": "zero", "episodes": 3, "successes": 0},
            ],
        }
        evaluation_path = root / "evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        return source_path, evaluation_path, source, evaluation

    def test_build_assigns_three_weight_classes_without_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path, evaluation_path, source, evaluation = self._fixtures(Path(tmp))
            result = builder.build_weighted_manifest(
                source,
                evaluation,
                source_path=source_path,
                evaluation_path=evaluation_path,
                full_weight=1.0,
                partial_weight=2.0,
                zero_weight=4.0,
            )
        self.assertEqual([motion["id"] for motion in result["motions"]], ["full", "partial", "zero"])
        self.assertEqual([motion["weight"] for motion in result["motions"]], [1.0, 2.0, 4.0])
        self.assertEqual(
            result["curriculum"]["class_counts"],
            {"full_success": 1, "partial_success": 1, "zero_success": 1},
        )
        self.assertEqual(result["curriculum"]["frames"], 600)

    def test_build_requires_full_ordered_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path, evaluation_path, source, evaluation = self._fixtures(Path(tmp))
            evaluation["selected_motion_indices"] = [0, 2]
            with self.assertRaisesRegex(ValueError, "every source motion"):
                builder.build_weighted_manifest(
                    source,
                    evaluation,
                    source_path=source_path,
                    evaluation_path=evaluation_path,
                    full_weight=1.0,
                    partial_weight=2.0,
                    zero_weight=4.0,
                )

    def test_build_rejects_manifest_hash_mismatch_and_bad_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_path, evaluation_path, source, evaluation = self._fixtures(Path(tmp))
            evaluation["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                builder.build_weighted_manifest(
                    source,
                    evaluation,
                    source_path=source_path,
                    evaluation_path=evaluation_path,
                    full_weight=1.0,
                    partial_weight=2.0,
                    zero_weight=4.0,
                )
            evaluation["manifest_sha256"] = builder.sha256(source_path)
            with self.assertRaisesRegex(ValueError, "zero >= partial >= full"):
                builder.build_weighted_manifest(
                    source,
                    evaluation,
                    source_path=source_path,
                    evaluation_path=evaluation_path,
                    full_weight=3.0,
                    partial_weight=2.0,
                    zero_weight=1.0,
                )


if __name__ == "__main__":
    unittest.main()
