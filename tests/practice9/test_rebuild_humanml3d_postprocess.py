from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "scripts/practice9/rebuild_humanml3d_postprocess.py"
)
SPEC = importlib.util.spec_from_file_location(
    "rebuild_humanml3d_postprocess", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)

OFFICIAL_ROOT = Path(
    os.environ.get(
        "HUMANML3D_OFFICIAL_ROOT",
        "/root/gpufree-data/datasets/HumanML3D-official",
    )
)


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source_path",
                "start_frame",
                "end_frame",
                "new_name",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_contract(
    segment_names: set[str],
    represent_names: set[str] | None = None,
) -> pipeline.IndexContract:
    return pipeline.IndexContract(
        rows=(),
        unique_sources=frozenset(),
        segment_names=frozenset(segment_names),
        represent_names=frozenset(
            segment_names
            if represent_names is None
            else represent_names
        ),
    )


def valid_skeleton_motion(frames: int = 6) -> np.ndarray:
    raw = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    chains = (
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    )
    rest = np.zeros((pipeline.JOINTS_NUM, 3), dtype=np.float32)
    for chain in chains:
        for index in range(1, len(chain)):
            parent, child = chain[index - 1], chain[index]
            length = 0.35 if child in (13, 14) else 0.25
            rest[child] = rest[parent] + raw[child] * length
    angle = np.float32(0.3)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )
    rest = rest @ rotation.T
    result = np.repeat(rest[None], frames, axis=0)
    result[..., 1] += 1.0
    result[:, :, 0] += np.arange(frames, dtype=np.float32)[:, None] * 0.01
    return result


class IndexAndPathTests(unittest.TestCase):
    def test_index_encodes_humanact_and_trim_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.csv"
            write_index(
                path,
                [
                    {
                        "source_path": (
                            "./pose_data/Eyes_Japan_Dataset/a_poses.npy"
                        ),
                        "start_frame": "2",
                        "end_frame": "6",
                        "new_name": "000000.npy",
                    },
                    {
                        "source_path": (
                            "./pose_data/humanact12/humanact12/b.npy"
                        ),
                        "start_frame": "0",
                        "end_frame": "-1",
                        "new_name": "000001.npy",
                    },
                ],
            )
            contract = pipeline.load_index_contract(
                path, enforce_official_counts=False
            )
            self.assertEqual(contract.rows[0].trim_frames, 60)
            self.assertEqual(contract.rows[0].output_frames, 4)
            self.assertTrue(contract.rows[1].humanact)
            self.assertIsNone(contract.rows[1].output_frames)
            self.assertEqual(len(contract.segment_names), 4)

    def test_path_rejects_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(pipeline.SafetyError):
                pipeline.checked_path(root, "../escape.npy")
            outside = root / "outside"
            outside.mkdir()
            output = root / "output"
            output.mkdir()
            (output / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(pipeline.SafetyError):
                pipeline.checked_path(output, "link/file.npy")
            leaf = output / "leaf.npy"
            leaf.symlink_to(outside / "leaf.npy")
            with self.assertRaises(pipeline.SafetyError):
                pipeline.checked_path(output, "leaf.npy")


class AtomicAndSegmentTests(unittest.TestCase):
    def test_atomic_npy_rejects_nonfinite_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "joints.npy"
            good = np.zeros((3, 22, 3), dtype=np.float32)
            pipeline.atomic_npy(
                path, good, pipeline.validate_joints_array
            )
            bad = good.copy()
            bad[0, 0, 0] = np.nan
            with self.assertRaises(pipeline.PipelineError):
                pipeline.atomic_npy(
                    path, bad, pipeline.validate_joints_array
                )
            self.assertTrue(np.isfinite(np.load(path)).all())

    def test_segment_trim_flip_and_exact_mirror(self) -> None:
        source = np.zeros((70, 22, 3), dtype=np.float32)
        source[..., 0] = np.arange(70, dtype=np.float32)[:, None]
        row = pipeline.IndexRow(
            source_rel="Eyes_Japan_Dataset/a.npy",
            start_frame=2,
            end_frame=6,
            new_name="000000.npy",
            humanact=False,
            trim_frames=60,
        )
        base, mirrored = pipeline.segment_positions(source, row)
        np.testing.assert_array_equal(
            base[:, 0, 0], -np.arange(62, 66, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            mirrored, pipeline.swap_left_right(base)
        )
        human = pipeline.IndexRow(
            source_rel="humanact12/humanact12/a.npy",
            start_frame=0,
            end_frame=-1,
            new_name="000001.npy",
            humanact=True,
            trim_frames=0,
        )
        extras = np.ones((4, 2, 3), dtype=np.float32)
        extras[..., 0] = 2.0
        human_source = np.concatenate([source[:4], extras], axis=1)
        human_base, human_mirror = pipeline.segment_positions(
            human_source, human
        )
        np.testing.assert_array_equal(human_base, human_source)
        self.assertEqual(human_base.shape, (4, 24, 3))
        np.testing.assert_array_equal(
            human_mirror[:, 22:, 0], -human_source[:, 22:, 0]
        )
        np.testing.assert_array_equal(
            human_mirror[:, 22:, 1:], human_source[:, 22:, 1:]
        )

    def test_state_resume_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_rel = "joints/a.npy"
            output = root / output_rel
            info = pipeline.atomic_npy(
                output,
                np.zeros((3, 22, 3), dtype=np.float32),
                pipeline.validate_joints_array,
            )
            with pipeline.StateDB(
                root / "postprocess_state.sqlite3", root
            ) as state:
                state.record(
                    "segment", output_rel, "fingerprint", output_rel, info
                )
                self.assertTrue(
                    state.complete(
                        "segment",
                        output_rel,
                        "fingerprint",
                        output_rel,
                        pipeline.validate_joints_array,
                        verify_hash=True,
                    )
                )
                pipeline.atomic_npy(
                    output,
                    np.ones((3, 22, 3), dtype=np.float32),
                    pipeline.validate_joints_array,
                )
                self.assertFalse(
                    state.complete(
                        "segment",
                        output_rel,
                        "fingerprint",
                        output_rel,
                        pipeline.validate_joints_array,
                        verify_hash=True,
                    )
                )

    def test_state_counts_use_read_only_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "postprocess_state.sqlite3"
            output_rel = "joints/a.npy"
            info = pipeline.atomic_npy(
                root / output_rel,
                np.zeros((3, 22, 3), dtype=np.float32),
                pipeline.validate_joints_array,
            )
            with pipeline.StateDB(state_path, root) as state:
                state.record(
                    "segment", output_rel, "fingerprint", output_rel, info
                )
            modified = state_path.stat().st_mtime_ns
            self.assertEqual(
                pipeline.read_state_counts(state_path), {"segment": 1}
            )
            self.assertEqual(state_path.stat().st_mtime_ns, modified)

    def test_pose_ledger_is_bound_to_success_record_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "dataset/a.npy"
            fingerprint = f"{'a' * 64}:item"
            digest = "b" * 64
            contract = pipeline.IndexContract(
                rows=(),
                unique_sources=frozenset({source}),
                segment_names=frozenset(),
                represent_names=frozenset(),
            )
            info = pipeline.NpyInfo(
                shape=(3, 22, 3),
                dtype="float32",
                size=128,
                sha256=digest,
            )
            with pipeline.StateDB(
                root / "state.sqlite3", root
            ) as state:
                state.record(
                    "pose", source, fingerprint, source, info
                )
            marker = {
                "pose_fingerprint": "a" * 64,
                "record_set_sha256": pipeline.canonical_sha256(
                    [
                        {
                            "source": source,
                            "fingerprint": fingerprint,
                            "output_sha256": digest,
                        }
                    ]
                ),
            }
            self.assertEqual(
                pipeline.load_pose_ledger(root, contract, marker),
                {source: digest},
            )
            marker["record_set_sha256"] = "c" * 64
            with self.assertRaises(pipeline.PipelineError):
                pipeline.load_pose_ledger(root, contract, marker)

    def test_state_rejects_main_and_sidecar_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.write_bytes(b"not a database")
            main = root / "main.sqlite3"
            main.symlink_to(outside)
            with self.assertRaises(pipeline.SafetyError):
                pipeline.StateDB(main, root)
            main.unlink()
            sidecar = Path(f"{main}-wal")
            sidecar.symlink_to(outside)
            with self.assertRaises(pipeline.SafetyError):
                pipeline.StateDB(main, root)

    def test_file_lock_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.lock"
            outside.write_text("", encoding="utf-8")
            lock = root / "lock"
            lock.symlink_to(outside)
            with self.assertRaises(
                pipeline.PipelineError
            ), pipeline.FileLock(lock, shared=True):
                self.fail("symlinked lock was acquired")

    def test_marker_invalidation_and_temp_cleanup_are_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / "manifests"
            manifests.mkdir()
            for name in pipeline.POSTPROCESS_MARKERS:
                (manifests / name).write_text("{}\n", encoding="utf-8")
            pipeline.invalidate_stage_markers(root, "represent")
            self.assertTrue(
                (
                    manifests / "postprocess-segment-success.json"
                ).is_file()
            )
            for name in pipeline.MARKER_INVALIDATION["represent"]:
                self.assertFalse((manifests / name).exists())

            outputs = root / "outputs"
            outputs.mkdir()
            orphan = outputs / ".a.npy.deadbeef.tmp"
            orphan.write_bytes(b"partial")
            removed = pipeline.cleanup_atomic_temps(
                outputs, {"a.npy"}
            )
            self.assertEqual(removed, (orphan.name,))
            self.assertFalse(orphan.exists())
            unknown = outputs / ".b.npy.deadbeef.tmp"
            unknown.write_bytes(b"unknown")
            with self.assertRaises(pipeline.SafetyError):
                pipeline.cleanup_atomic_temps(outputs, {"a.npy"})
            self.assertTrue(unknown.exists())
            unknown.unlink()
            outside = root / "outside.tmp"
            outside.write_bytes(b"keep")
            unsafe = outputs / ".a.npy.symlink.tmp"
            unsafe.symlink_to(outside)
            with self.assertRaises(pipeline.SafetyError):
                pipeline.cleanup_atomic_temps(outputs, {"a.npy"})
            self.assertEqual(outside.read_bytes(), b"keep")


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pipeline.force_cpu_environment()
        cls.backend = pipeline.load_official_backend(OFFICIAL_ROOT)

    def test_official_backend_is_explicit_and_cpu(self) -> None:
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(
            Path(
                sys.modules["common.skeleton"].__file__
            ).resolve(),
            (OFFICIAL_ROOT / "common/skeleton.py").resolve(),
        )

    def test_reference_comparison_reports_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.npy"
            value = np.zeros((0, 263), dtype=np.float32)
            np.save(path, value, allow_pickle=False)
            result = pipeline.compare_reference_array(
                value,
                path,
                lambda _array, **_kwargs: None,
                rtol=0.0,
                atol=0.0,
            )
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["mean_abs"], 0.0)
        self.assertTrue(result["allclose"])

    def test_normal_representation_is_finite_263d(self) -> None:
        motion = valid_skeleton_motion(8)
        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.npy"
            np.save(target_path, motion, allow_pickle=False)
            offsets = pipeline.target_offsets_from_file(
                target_path, self.backend
            )
            joints, vectors, diagnostics = pipeline.process_motion(
                motion, offsets, self.backend
            )
        self.assertEqual(joints.shape, (7, 22, 3))
        self.assertEqual(vectors.shape, (7, 263))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertTrue(np.isfinite(joints).all())
        self.assertTrue(np.isfinite(vectors).all())
        self.assertFalse(any(diagnostics.values()))

    def test_repair_path_makes_degenerate_motion_finite(self) -> None:
        target = valid_skeleton_motion(6)
        with tempfile.TemporaryDirectory() as temporary:
            target_path = Path(temporary) / "target.npy"
            np.save(target_path, target, allow_pickle=False)
            offsets = pipeline.target_offsets_from_file(
                target_path, self.backend
            )
            degenerate = np.zeros((5, 22, 3), dtype=np.float32)
            with self.assertRaises(pipeline.PipelineError):
                pipeline.process_motion(
                    degenerate, offsets, self.backend, repair=False
                )
            joints, vectors, diagnostics = pipeline.process_motion(
                degenerate, offsets, self.backend, repair=True
            )
        self.assertTrue(np.isfinite(joints).all())
        self.assertTrue(np.isfinite(vectors).all())
        self.assertGreater(diagnostics["unit_fallbacks"], 0)

    def test_small_representation_stage_repairs_only_known_ids(self) -> None:
        names = {
            pipeline.TARGET_SKELETON_ID,
            "000022.npy",
            "000023.npy",
            "007975.npy",
            "M007975.npy",
            "009707.npy",
        }
        represented = names - {"009707.npy"}
        contract = make_contract(names, represented)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joints_root = root / "joints"
            joints_root.mkdir(parents=True)
            normal = valid_skeleton_motion(6)
            for name in (
                pipeline.TARGET_SKELETON_ID,
                "000022.npy",
            ):
                np.save(joints_root / name, normal, allow_pickle=False)
            normal_24 = np.concatenate(
                [
                    normal,
                    np.full(
                        (normal.shape[0], 2, 3),
                        123.0,
                        dtype=np.float32,
                    ),
                ],
                axis=1,
            )
            np.save(
                joints_root / "000023.npy",
                normal_24,
                allow_pickle=False,
            )
            degenerate = np.zeros((5, 22, 3), dtype=np.float32)
            for name in ("007975.npy", "M007975.npy"):
                np.save(
                    joints_root / name,
                    degenerate,
                    allow_pickle=False,
                )
            np.save(
                joints_root / "009707.npy",
                normal[:1],
                allow_pickle=False,
            )
            with pipeline.StateDB(
                root / "postprocess_state.sqlite3", root
            ) as state:
                result = pipeline.represent_dataset(
                    contract,
                    OFFICIAL_ROOT,
                    root,
                    state,
                    "a" * 64,
                    pipeline.SpaceGuard(root, 0),
                    verify_resume_hash=True,
                    enforce_official_counts=False,
                    compare_reference=False,
                )
            vectors_22 = np.load(
                root / "HumanML3D/new_joint_vecs/000022.npy"
            )
            vectors_24 = np.load(
                root / "HumanML3D/new_joint_vecs/000023.npy"
            )
        self.assertEqual(result["files"], 5)
        np.testing.assert_array_equal(vectors_22, vectors_24)
        self.assertEqual(
            set(result["repaired_nonfinite_ids"]),
            {"007975.npy", "M007975.npy"},
        )
        self.assertTrue(result["finite"])


class StatisticsTests(unittest.TestCase):
    def test_dual_statistics_exclude_repairs_only_from_primary(self) -> None:
        names = {
            "000100.npy",
            "007975.npy",
            "M007975.npy",
        }
        contract = make_contract(names)
        generator = np.random.default_rng(42)
        values = {
            "000100.npy": generator.normal(
                0.0, 1.0, size=(6, 263)
            ).astype(np.float32),
            "007975.npy": generator.normal(
                2.0, 1.0, size=(5, 263)
            ).astype(np.float32),
            "M007975.npy": generator.normal(
                -2.0, 1.0, size=(5, 263)
            ).astype(np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors_root = root / "HumanML3D/new_joint_vecs"
            vectors_root.mkdir(parents=True)
            for name, array in values.items():
                np.save(vectors_root / name, array, allow_pickle=False)
            with pipeline.StateDB(
                root / "postprocess_state.sqlite3", root
            ) as state:
                result = pipeline.stats_dataset(
                    contract,
                    OFFICIAL_ROOT,
                    root,
                    state,
                    "b" * 64,
                    pipeline.SpaceGuard(root, 0),
                    verify_resume_hash=True,
                    enforce_official_counts=False,
                    compare_reference=False,
                )
            primary_mean = np.load(root / "HumanML3D/Mean.npy")
            stable_mean = np.load(
                root / "HumanML3D/Mean_stable.npy"
            )
            all_mean = np.load(
                root / "HumanML3D/Mean_all_finite.npy"
            )
            primary_std = np.load(root / "HumanML3D/Std.npy")
            stable_std = np.load(
                root / "HumanML3D/Std_stable.npy"
            )
            finite_ids = set(
                (
                    root / "HumanML3D/finite_ids.txt"
                ).read_text(encoding="utf-8").split()
            )
            excluded = set(
                (
                    root
                    / "HumanML3D/compat_stats_excluded_ids.txt"
                ).read_text(encoding="utf-8").split()
            )
            (root / "HumanML3D/finite_ids.txt").write_text(
                "M007975\n007975\n000100\n",
                encoding="utf-8",
            )
            with pipeline.StateDB(
                root / "postprocess_state.sqlite3", root
            ) as state, self.assertRaises(pipeline.PipelineError):
                pipeline.verify_statistics_files(
                    contract,
                    OFFICIAL_ROOT,
                    root,
                    state,
                    "b" * 64,
                    verify_hash=True,
                    compare_reference=False,
                )
        np.testing.assert_allclose(
            primary_mean,
            values["000100.npy"].mean(axis=0),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            stable_mean,
            values["000100.npy"].mean(
                axis=0, dtype=np.float64
            ).astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(np.array_equal(primary_mean, all_mean))
        for array in (primary_std, stable_std):
            for start, end in pipeline.FEATURE_GROUPS:
                self.assertTrue(
                    np.all(array[start:end] == array[start])
                )
        self.assertEqual(
            finite_ids, {"000100", "007975", "M007975"}
        )
        self.assertEqual(excluded, {"007975", "M007975"})
        self.assertEqual(result["primary_compatibility"]["files"], 1)
        self.assertEqual(
            result["primary_compatibility"]["order"], "os.listdir"
        )
        self.assertEqual(result["stable_compatibility"]["files"], 1)
        self.assertEqual(result["all_finite"]["files"], 3)
        self.assertTrue(result["finite"])

    def test_legacy_memory_gate_rejects_insufficient_ram(self) -> None:
        required = (
            1024 * pipeline.LEGACY_RAM_MULTIPLIER
            + pipeline.LEGACY_RAM_SAFETY_BYTES
        )
        with self.assertRaises(pipeline.PipelineError):
            pipeline.legacy_memory_gate(
                1024, available_bytes=required - 1
            )


class MomentTests(unittest.TestCase):
    def test_online_moments_match_population_statistics(self) -> None:
        generator = np.random.default_rng(7)
        first = generator.normal(size=(7, 263)).astype(np.float32)
        second = generator.normal(size=(9, 263)).astype(np.float32)
        moments = pipeline.OnlineMoments(263)
        moments.update(first)
        moments.update(second)
        mean, std = moments.finalize()
        combined = np.concatenate([first, second]).astype(np.float64)
        expected_std = combined.std(axis=0).astype(np.float32)
        for start, end in pipeline.FEATURE_GROUPS:
            expected_std[start:end] = expected_std[start:end].mean(
                dtype=np.float32
            )
        np.testing.assert_allclose(
            mean, combined.mean(axis=0), rtol=1.0e-6, atol=1.0e-6
        )
        np.testing.assert_allclose(
            std, expected_std, rtol=1.0e-6, atol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
