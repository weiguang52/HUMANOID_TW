from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[2] / "scripts/practice9/select_humanml3d_locomotion.py"
SPEC = importlib.util.spec_from_file_location("selector", SCRIPT)
assert SPEC and SPEC.loader
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


class SelectorTests(unittest.TestCase):
    def test_geometry_and_classification(self) -> None:
        joints = np.zeros((10, 22, 3), dtype=np.float32)
        joints[:, 0, 0] = np.linspace(0, 1, 10)
        joints[:, 1, 0] = -0.1
        joints[:, 2, 0] = 0.1
        displacement, path, yaw = selector.geometry(joints)
        self.assertAlmostEqual(displacement, 1.0)
        self.assertAlmostEqual(path, 1.0)
        self.assertEqual(selector.classify("walks backwards", displacement, yaw), "backward")

    def test_inspect_rejects_non_locomotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = np.zeros((50, 22, 3), dtype=np.float32)
            motion[:, 0, 0] = np.linspace(0, 1, 50)
            motion[:, 1, 0] = -0.1
            motion[:, 2, 0] = 0.1
            source = root / "000001.npy"
            np.save(source, motion)
            text = root / "000001.txt"
            text.write_text("a person runs and jumps#tokens#0#0\n", encoding="utf-8")
            self.assertIsNone(selector.inspect(source, text, 40, 300))
            text.write_text("a person walks forward#tokens#0#0\n", encoding="utf-8")
            self.assertIsNotNone(selector.inspect(source, text, 40, 300))

    def test_balanced_is_deterministic(self) -> None:
        items = [
            selector.Candidate(f"{i:06d}", "x", "walk", 50, 1.0, float(i + 1), 0.0, "forward", float(i + 1))
            for i in range(10)
        ]
        first = selector.select_balanced(items, 4)
        second = selector.select_balanced(list(reversed(items)), 4)
        self.assertEqual([x.motion_id for x in first], [x.motion_id for x in second])
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
