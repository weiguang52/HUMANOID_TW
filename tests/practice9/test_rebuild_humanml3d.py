from __future__ import annotations

import csv
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts/practice9/rebuild_humanml3d.py"
SPEC = importlib.util.spec_from_file_location(
    "rebuild_humanml3d", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


class PathSafetyTests(unittest.TestCase):
    def test_normalize_member_rejects_escape_forms(self) -> None:
        self.assertEqual(
            pipeline.normalize_member_name("./ACCAD/a.npz"),
            "ACCAD/a.npz",
        )
        for unsafe in (
            "",
            "../a",
            "ACCAD/../a",
            "/ACCAD/a",
            "ACCAD\\a",
        ):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(pipeline.SafetyError):
                    pipeline.normalize_member_name(unsafe)

    def test_checked_output_path_is_below_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                pipeline.checked_output_path(root, "KIT/a.npy"),
                root.resolve() / "KIT/a.npy",
            )
            with self.assertRaises(pipeline.SafetyError):
                pipeline.checked_output_path(root, "../escape.npy")

    def test_production_roots_must_be_disjoint_on_data_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            source = data / "official"
            archives = data / "archives"
            staging = data / "staging"
            source.mkdir()
            archives.mkdir()
            with mock.patch.object(pipeline, "DATA_ROOT", data):
                pipeline.validate_layout(source, archives, staging, data)
                with self.assertRaises(pipeline.SafetyError):
                    pipeline.validate_layout(source, archives, source / "out", data)


class ArchiveSafetyTests(unittest.TestCase):
    @staticmethod
    def make_tar(
        path: Path, entries: list[tuple[str, bytes, str]]
    ) -> None:
        with tarfile.open(path, "w:bz2") as archive:
            for name, payload, kind in entries:
                info = tarfile.TarInfo(name)
                if kind == "file":
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                elif kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = "target"
                    archive.addfile(info)
                else:
                    raise AssertionError(kind)

    def test_tar_scan_enforces_mapping_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "KIT.tar.bz2"
            self.make_tar(
                path,
                [
                    ("LICENSE.txt", b"license", "file"),
                    ("KIT/1/a.npz", b"payload", "file"),
                ],
            )
            scan = pipeline.scan_tar_archive(
                path, "KIT", "KIT", {"KIT/1/a.npz"}
            )
            self.assertEqual(scan.required_members, 1)
            self.assertEqual(scan.regular_files, 2)
            self.assertEqual(scan.expanded_bytes, 14)
            self.assertEqual(len(scan.sha256), 64)

    def test_tar_scan_rejects_unlisted_root_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "KIT.tar.bz2"
            self.make_tar(path, [("README.txt", b"x", "file")])
            with self.assertRaises(pipeline.SafetyError):
                pipeline.scan_tar_archive(path, "KIT", "KIT", set())

    def test_tar_scan_rejects_traversal_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traversal = root / "traversal.tar.bz2"
            self.make_tar(
                traversal, [("../escape", b"x", "file")]
            )
            with self.assertRaises(pipeline.SafetyError):
                pipeline.scan_tar_archive(
                    traversal, "KIT", "KIT", set()
                )
            linked = root / "linked.tar.bz2"
            self.make_tar(
                linked, [("KIT/link", b"", "symlink")]
            )
            with self.assertRaises(pipeline.SafetyError):
                pipeline.scan_tar_archive(
                    linked, "KIT", "KIT", set()
                )

    def test_zip_scan_rejects_wrong_top_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / "wrong.zip"
            with zipfile.ZipFile(wrong, "w") as archive:
                archive.writestr("other/a.npy", b"x")
            with self.assertRaises(pipeline.SafetyError):
                pipeline.scan_humanact_zip(wrong, set())

            linked = root / "linked.zip"
            with zipfile.ZipFile(linked, "w") as archive:
                info = zipfile.ZipInfo(
                    "humanact12/humanact12/link"
                )
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "target")
            with self.assertRaises(pipeline.SafetyError):
                pipeline.scan_humanact_zip(linked, set())

    def test_mapping_has_all_eighteen_canonical_tops(self) -> None:
        self.assertEqual(len(pipeline.ARCHIVE_TOPS), 18)
        self.assertEqual(
            pipeline.ARCHIVE_TOPS["BMLrub"],
            "BioMotionLab_NTroje",
        )
        self.assertEqual(
            pipeline.ARCHIVE_TOPS["Transitions"],
            "Transitions_mocap",
        )


class IndexContractTests(unittest.TestCase):
    def test_maps_sources_to_tar_and_humanact_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.csv"
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
                writer.writerow(
                    {
                        "source_path": "./pose_data/KIT/1/a_poses.npy",
                        "start_frame": "0",
                        "end_frame": "10",
                        "new_name": "000001.npy",
                    }
                )
                writer.writerow(
                    {
                        "source_path": (
                            "./pose_data/humanact12/"
                            "humanact12/b.npy"
                        ),
                        "start_frame": "-1",
                        "end_frame": "-1",
                        "new_name": "000002.npy",
                    }
                )
            contract = pipeline.load_index_contract(
                path, enforce_official_counts=False
            )
            self.assertEqual(
                contract.required_by_archive["KIT"],
                {"KIT/1/a_poses.npz"},
            )
            self.assertEqual(
                contract.required_humanact,
                {"humanact12/humanact12/b.npy"},
            )


    def test_fingerprints_bind_models_and_index(self) -> None:
        models = {
            relative: {"sha256": "a" * 64}
            for relative in pipeline.MODEL_FILES
        }
        original = pipeline.build_algorithm_contract(models)
        changed = {
            name: dict(metadata)
            for name, metadata in models.items()
        }
        changed[pipeline.MODEL_FILES[0]]["sha256"] = "b" * 64
        modified = pipeline.build_algorithm_contract(changed)
        self.assertNotEqual(
            original["fingerprint"], modified["fingerprint"]
        )
        self.assertNotEqual(
            pipeline.build_pose_fingerprint(original["fingerprint"], "c" * 64),
            pipeline.build_pose_fingerprint(original["fingerprint"], "d" * 64),
        )


class DurablePoseTests(unittest.TestCase):
    def test_atomic_write_rejects_nonfinite_without_damage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pose.npy"
            good = np.zeros((2, 22, 3), dtype=np.float32)
            pipeline.save_npy_atomic(path, good)
            self.assertEqual(
                pipeline.validate_pose_file(path),
                ((2, 22, 3), "float32"),
            )
            bad = good.copy()
            bad[0, 0, 0] = np.nan
            with self.assertRaises(pipeline.PipelineError):
                pipeline.save_npy_atomic(path, bad)
            self.assertTrue(np.isfinite(np.load(path)).all())

    def test_resume_ledger_detects_output_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pose.npy"
            pipeline.save_npy_atomic(
                output, np.zeros((2, 22, 3), dtype=np.float32)
            )
            with pipeline.StateDB(root / "state.sqlite3") as state:
                state.record(
                    "pose",
                    "pose.npy",
                    "source-v1",
                    "pose.npy",
                    output,
                    (2, 22, 3),
                    "float32",
                )
                self.assertTrue(
                    state.complete(
                        "pose",
                        "pose.npy",
                        "source-v1",
                        output,
                        verify_hash=True,
                    )
                )
                pipeline.save_npy_atomic(
                    output,
                    np.ones((2, 22, 3), dtype=np.float32),
                )
                self.assertFalse(
                    state.complete(
                        "pose",
                        "pose.npy",
                        "source-v1",
                        output,
                        verify_hash=True,
                    )
                )

    def test_humanact_exact_two_level_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zip_path = root / "humanact12.zip"
            member = "humanact12/humanact12/sample.npy"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    member,
                    npy_bytes(
                        np.zeros((3, 24, 3), dtype=np.float32)
                    ),
                )
            pose_root = root / "pose_data"
            with pipeline.StateDB(root / "state.sqlite3") as state:
                count = pipeline.materialize_humanact(
                    zip_path,
                    {member},
                    pose_root,
                    state,
                    "zip-v1",
                    {member: 1},
                    pipeline.SpaceGuard(root, 0),
                    verify_resume_hash=True,
                )
            self.assertEqual(count, 1)
            self.assertTrue(
                (
                    pose_root
                    / "humanact12/humanact12/sample.npy"
                ).is_file()
            )
            self.assertFalse(
                (
                    pose_root
                    / "humanact12/humanact12/humanact12"
                ).exists()
            )


    def test_amass_loader_enforces_finite_schema(self) -> None:
        poses = np.zeros((4, 156), dtype=np.float32)
        def record(values: np.ndarray) -> io.BytesIO:
            stream = io.BytesIO()
            np.savez(
                stream,
                mocap_framerate=np.array(40.0, dtype=np.float32),
                trans=np.zeros((4, 3), dtype=np.float32),
                poses=values,
                gender=np.array("male"),
                betas=np.zeros(16, dtype=np.float32),
            )
            stream.seek(0)
            return stream

        stride, loaded, *_ = pipeline.load_amass_record(record(poses), "ok")
        self.assertEqual(stride, 2)
        self.assertEqual(loaded.shape, (4, 156))
        poses[0, 0] = np.nan
        with self.assertRaises(pipeline.PipelineError):
            pipeline.load_amass_record(record(poses), "bad")


    def test_space_guard_enforces_continuous_reserve(self) -> None:
        usage = mock.Mock(free=1024)
        with mock.patch.object(pipeline.shutil, "disk_usage", return_value=usage):
            guard = pipeline.SpaceGuard(Path("/unused"), 512)
            guard.require(extra_bytes=512, label="within reserve")
            with self.assertRaises(pipeline.PipelineError):
                guard.require(extra_bytes=513, label="over reserve")


    def test_verify_requires_exact_files_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose_root = root / "pose_data"
            relative = "KIT/1/a_poses.npy"
            output = pipeline.checked_output_path(pose_root, relative)
            array = np.zeros((3, 22, 3), dtype=np.float32)
            pipeline.save_npy_atomic(output, array)
            contract = pipeline.IndexContract(
                rows=(),
                unique_sources=frozenset({relative}),
                minimum_frames={relative: 3},
                required_by_archive={},
                required_humanact=frozenset(),
            )
            pose_fingerprint = "a" * 64
            with pipeline.StateDB(root / "state.sqlite3") as state:
                state.record(
                    "pose",
                    relative,
                    f"{pose_fingerprint}:source",
                    relative,
                    output,
                    array.shape,
                    str(array.dtype),
                )
                result = pipeline.verify_pose_tree(
                    contract,
                    pose_root,
                    state,
                    pose_fingerprint,
                    "b" * 64,
                    {"type": "cpu"},
                    verify_hash=True,
                )
                self.assertEqual(result["db_records"], 1)
                extra = pose_root / "extra.npy"
                pipeline.save_npy_atomic(extra, array)
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.verify_pose_tree(
                        contract,
                        pose_root,
                        state,
                        pose_fingerprint,
                        "b" * 64,
                        {"type": "cpu"},
                        verify_hash=True,
                    )


class DeviceGuardTests(unittest.TestCase):
    def test_cpu_guard_clears_visibility(self) -> None:
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        try:
            pipeline.force_cpu_only_environment()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("torch", pipeline.sys.modules)
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def test_gpu_parser_tracks_compute_processes(self) -> None:
        snapshots = pipeline.parse_gpu_snapshots(
            "0, GPU-a, 12, 0\n1, GPU-b, 300, 4\n",
            "GPU-b, 1234, 256\n",
        )
        self.assertEqual(
            snapshots["GPU-a"].compute_pids, frozenset()
        )
        self.assertEqual(
            snapshots["GPU-b"].compute_pids,
            frozenset({1234}),
        )


    def test_nspid_parser_includes_namespace_ids(self) -> None:
        status = "Name:\tpython\nNSpid:\t252868\t380840\n"
        self.assertEqual(
            pipeline.parse_nspid_status(status, 380840),
            frozenset({252868, 380840}),
        )

    def test_nspid_missing_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-status"
            self.assertEqual(
                pipeline.current_process_ids(missing, fallback_pid=77),
                frozenset({77}),
            )
        self.assertEqual(
            pipeline.parse_nspid_status("Name:\tpython\n", 88),
            frozenset({88}),
        )


    def test_gpu_guard_requires_exact_owned_pid(self) -> None:
        owned = 252868
        valid = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset({owned})
        )
        invalid = (
            pipeline.GpuSnapshot("1", "GPU-a", 128, 4, frozenset()),
            pipeline.GpuSnapshot("1", "GPU-a", 128, 4, frozenset({owned, 9})),
            pipeline.GpuSnapshot("1", "GPU-a", 128, 4, frozenset({999999})),
        )
        lease = pipeline.GpuLease(None, "GPU-a", Path("/unused"), 256, 5)
        lease.snapshot = valid
        lease.owned_compute_pid = owned
        with mock.patch.object(
            pipeline, "query_gpu_snapshots", return_value={"GPU-a": valid}
        ):
            lease.assert_no_foreign_compute()
        for observed in invalid:
            with self.subTest(observed=observed.compute_pids):
                with mock.patch.object(
                    pipeline, "query_gpu_snapshots", return_value={"GPU-a": observed}
                ):
                    with self.assertRaises(pipeline.PipelineError):
                        lease.assert_no_foreign_compute()


    @staticmethod
    def make_binding_lease():
        lease = pipeline.GpuLease(None, "GPU-a", Path("/unused"), 256, 5)
        lease.snapshot = pipeline.GpuSnapshot(
            "1", "GPU-a", 2, 0, frozenset()
        )
        lease.baseline_compute_pids = frozenset()
        lease.lock = mock.Mock()
        torch_module = mock.Mock()
        torch_module.cuda.device_count.return_value = 1
        return lease, torch_module

    def test_gpu_bind_accepts_unique_stable_host_pid(self) -> None:
        lease, torch_module = self.make_binding_lease()
        observed = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset({252868})
        )
        with mock.patch.object(
            pipeline, "current_process_ids", return_value=frozenset({380840})
        ), mock.patch.object(
            pipeline,
            "query_gpu_snapshots",
            side_effect=[{"GPU-a": observed}, {"GPU-a": observed}],
        ), mock.patch.object(
            pipeline.time, "sleep"
        ), mock.patch("builtins.print"):
            lease.bind_cuda_process(torch_module)
        self.assertEqual(lease.owned_compute_pid, 252868)
        torch_module.empty.assert_called_once_with(1, device="cuda:0")
        torch_module.cuda.synchronize.assert_called_once_with()

    def test_gpu_bind_rejects_two_pids(self) -> None:
        lease, torch_module = self.make_binding_lease()
        observed = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset({252868, 380840})
        )
        with mock.patch.object(
            pipeline, "current_process_ids", return_value=frozenset({252868, 380840})
        ), mock.patch.object(
            pipeline,
            "query_gpu_snapshots",
            side_effect=[{"GPU-a": observed}, {"GPU-a": observed}],
        ), mock.patch.object(
            pipeline.time, "sleep"
        ):
            with self.assertRaises(pipeline.PipelineError):
                lease.bind_cuda_process(torch_module)

    def test_gpu_bind_rejects_empty_process_set(self) -> None:
        lease, torch_module = self.make_binding_lease()
        observed = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset()
        )
        with mock.patch.object(
            pipeline, "current_process_ids", return_value=frozenset({380840})
        ), mock.patch.object(
            pipeline,
            "query_gpu_snapshots",
            side_effect=[{"GPU-a": observed}, {"GPU-a": observed}],
        ), mock.patch.object(
            pipeline.time, "sleep"
        ):
            with self.assertRaises(pipeline.PipelineError):
                lease.bind_cuda_process(torch_module)

    def test_gpu_bind_rejects_pid_change(self) -> None:
        lease, torch_module = self.make_binding_lease()
        first = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset({252868})
        )
        second = pipeline.GpuSnapshot(
            "1", "GPU-a", 128, 4, frozenset({252869})
        )
        with mock.patch.object(
            pipeline, "current_process_ids", return_value=frozenset({380840})
        ), mock.patch.object(
            pipeline,
            "query_gpu_snapshots",
            side_effect=[{"GPU-a": first}, {"GPU-a": second}],
        ), mock.patch.object(
            pipeline.time, "sleep"
        ):
            with self.assertRaises(pipeline.PipelineError):
                lease.bind_cuda_process(torch_module)


if __name__ == "__main__":
    unittest.main()
