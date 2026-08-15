#!/usr/bin/env python3
"""Safe, resumable CPU post-processing for a rebuilt HumanML3D pose cache.

This file is deliberately independent from rebuild_humanml3d.py.  It only
consumes that pipeline's verified pose cache and never writes to the official
source tree.  Its stages are:

segment -> represent -> stats -> verify -> ready.

All production outputs live below the data-disk staging root.  Every file is
written by same-directory replace, every stage has a durable success marker,
and the per-file resume ledger is separate from the pose pipeline's database.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

SCHEMA_VERSION = 1
POSE_SCHEMA_VERSION = 2
ALGORITHM_REVISION = "humanml3d-postprocess-v1"
FROZEN_UPSTREAM_SCRIPT_SHA256 = (
    "62de40ce9265d09fc49429c47c59438cf9ca7ff666d2e82ac60fedf5371ce0cd"
)
STATISTICS_ALGORITHM_REVISION = "humanml3d-statistics-v2"
DATA_ROOT = Path("/root/gpufree-data")
SOURCE_ROOT = DATA_ROOT / "datasets/HumanML3D-official"
STAGING_ROOT = (
    DATA_ROOT / "datasets/practice9/humanml3d_rebuild/staging-v1"
)

TARGET_FPS = 20
JOINTS_NUM = 22
HUMANACT_JOINTS = 24
FEATURE_DIM = 263
LEGACY_MEAN_REFERENCE_ATOL = 1.0e-3
LEGACY_STD_REFERENCE_ATOL = 2.0e-4
LEGACY_RAM_MULTIPLIER = 3
LEGACY_RAM_SAFETY_BYTES = 2 * 1024**3
OFFICIAL_INDEX_ROWS = 14_616
OFFICIAL_POSE_SOURCES = 11_715
OFFICIAL_SEGMENT_FILES = 29_232
OFFICIAL_REPRESENT_FILES = 29_228
OFFICIAL_REPRESENT_FRAMES = 4_117_392
OFFICIAL_COMPAT_STATS_FILES = 29_226
OFFICIAL_COMPAT_STATS_FRAMES = 4_117_224
TARGET_SKELETON_ID = "000021.npy"
REFERENCE_ID = "012314.npy"

EXPECTED_SHORT_IDS = frozenset(
    {
        "009707.npy",
        "M009707.npy",
        "011059.npy",
        "M011059.npy",
    }
)
REPAIRED_NONFINITE_IDS = frozenset(
    {"007975.npy", "M007975.npy"}
)
REPAIR_EPSILON = 1.0e-8

SOURCE_TRIM_FRAMES = {
    "Eyes_Japan_Dataset": 3 * TARGET_FPS,
    "MPI_HDM05": 3 * TARGET_FPS,
    "TotalCapture": TARGET_FPS,
    "MPI_Limits": TARGET_FPS,
    "Transitions_mocap": TARGET_FPS // 2,
}

RIGHT_CHAIN = [2, 5, 8, 11, 14, 17, 19, 21]
LEFT_CHAIN = [1, 4, 7, 10, 13, 16, 18, 20]
L_IDX1, L_IDX2 = 5, 8
FID_R, FID_L = [8, 11], [7, 10]
FACE_JOINT_INDEX = [2, 1, 17, 16]
FEET_THRESHOLD = 0.002

FEATURE_GROUPS = (
    (0, 1),
    (1, 3),
    (3, 4),
    (4, 67),
    (67, 193),
    (193, 259),
    (259, 263),
)

REFERENCE_HASHES = {
    "new_joints": (
        "739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b"
    ),
    "new_joint_vecs": (
        "6e6fc0204ea9e48abdf94f84e8be22b19a1f1e955c7b98e025b1d702f6fe96a0"
    ),
    "Mean": (
        "26e136555dab04c94a129d446c26e6b9939cbf045fbf77bcf5462c1fb5a2001c"
    ),
    "Std": (
        "6565a65ed9b31e23c328829a309e1c482be8b85fd23b43d65451a9b19a917f40"
    ),
}

POSTPROCESS_MARKERS = frozenset(
    {
        "postprocess-segment-success.json",
        "postprocess-represent-success.json",
        "postprocess-stats-success.json",
        "postprocess-verify-success.json",
        "postprocess-ready.json",
    }
)
MARKER_INVALIDATION = {
    "segment": (
        "postprocess-ready.json",
        "postprocess-verify-success.json",
        "postprocess-stats-success.json",
        "postprocess-represent-success.json",
        "postprocess-segment-success.json",
    ),
    "represent": (
        "postprocess-ready.json",
        "postprocess-verify-success.json",
        "postprocess-stats-success.json",
        "postprocess-represent-success.json",
    ),
    "stats": (
        "postprocess-ready.json",
        "postprocess-verify-success.json",
        "postprocess-stats-success.json",
    ),
    "verify": (
        "postprocess-ready.json",
        "postprocess-verify-success.json",
    ),
    "ready": (
        "postprocess-ready.json",
        "postprocess-verify-success.json",
    ),
}


class PipelineError(RuntimeError):
    """A controlled post-processing failure."""


class SafetyError(PipelineError):
    """A path, input, or integrity boundary was violated."""


@dataclasses.dataclass(frozen=True)
class IndexRow:
    source_rel: str
    start_frame: int
    end_frame: int
    new_name: str
    humanact: bool
    trim_frames: int

    @property
    def output_frames(self) -> int | None:
        if self.humanact:
            return None
        return self.end_frame - self.start_frame

    @property
    def segment_joints(self) -> int:
        return HUMANACT_JOINTS if self.humanact else JOINTS_NUM


@dataclasses.dataclass(frozen=True)
class IndexContract:
    rows: tuple[IndexRow, ...]
    unique_sources: frozenset[str]
    segment_names: frozenset[str]
    represent_names: frozenset[str]


@dataclasses.dataclass(frozen=True)
class NpyInfo:
    shape: tuple[int, ...]
    dtype: str
    size: int
    sha256: str


def import_numpy():
    import numpy as np

    return np


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PipelineError(f"{label}: invalid SHA-256")
    return value


def normalize_relative(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise SafetyError(f"unsafe relative path: {name!r}")
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise SafetyError(f"unsafe relative path: {name!r}")
    return path.as_posix()


def checked_path(root: Path, relative: str) -> Path:
    normalized = normalize_relative(relative)
    resolved_root = root.resolve()
    lexical = resolved_root.joinpath(*PurePosixPath(normalized).parts)
    resolved_candidate = lexical.resolve()
    if resolved_candidate == resolved_root or not resolved_candidate.is_relative_to(
        resolved_root
    ):
        raise SafetyError(f"path escapes root: {relative!r}")
    current = resolved_root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise SafetyError(f"symlink in output path: {current}")
    return lexical


def reject_symlink_components(path: Path) -> None:
    current = Path(os.path.abspath(path))
    for candidate in (current, *current.parents):
        if candidate.is_symlink():
            raise SafetyError(f"symlink in writable path: {candidate}")


def atomic_json(path: Path, payload: object) -> None:
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_text(path: Path, text: str) -> None:
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def manifest_path(staging_root: Path, name: str) -> Path:
    if name != "pose-success.json" and name not in POSTPROCESS_MARKERS:
        raise SafetyError(f"unknown manifest name: {name!r}")
    return checked_path(staging_root, f"manifests/{name}")


def invalidate_stage_markers(staging_root: Path, stage: str) -> None:
    names = MARKER_INVALIDATION.get(stage)
    if names is None:
        raise PipelineError(f"unknown invalidation stage: {stage}")
    manifest_root = checked_path(staging_root, "manifests")
    manifest_root.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temps(manifest_root, set(POSTPROCESS_MARKERS))
    changed = False
    for name in names:
        path = manifest_path(staging_root, name)
        if not path.exists():
            continue
        if not path.is_file():
            raise SafetyError(f"manifest is not a regular file: {path}")
        path.unlink()
        changed = True
    if changed:
        fsync_directory(manifest_root)


def cleanup_atomic_temps(
    root: Path, expected_targets: set[str]
) -> tuple[str, ...]:
    """Remove only this writer's recognizable orphan temp files."""
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise SafetyError(f"unsafe temporary cleanup root: {root}")
    removed: list[str] = []
    for candidate in root.iterdir():
        name = candidate.name
        if (
            not name.startswith(".")
            or not name.endswith(".tmp")
        ):
            continue
        body = name[1:-4]
        target, separator, token = body.rpartition(".")
        if not separator or not token or target not in expected_targets:
            raise SafetyError(
                f"unknown atomic temp artifact: {candidate}"
            )
        if candidate.is_symlink() or not candidate.is_file():
            raise SafetyError(f"unsafe atomic temp artifact: {candidate}")
        candidate.unlink()
        removed.append(name)
    if removed:
        fsync_directory(root)
    return tuple(sorted(removed))


def source_relative(value: str) -> str:
    prefix = "./pose_data/"
    if not value.startswith(prefix):
        raise SafetyError(f"index source outside pose_data: {value!r}")
    return normalize_relative(value[len(prefix) :])


def parse_frame(
    value: str,
    field: str,
    row_number: int,
    *,
    allow_minus_one: bool,
) -> int:
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError(
            f"index row {row_number}: invalid {field}={value!r}"
        ) from exc
    minimum = -1 if allow_minus_one else 0
    if numeric != integer or integer < minimum:
        raise PipelineError(
            f"index row {row_number}: invalid {field}={value!r}"
        )
    return integer


def load_index_contract(
    path: Path, *, enforce_official_counts: bool = True
) -> IndexContract:
    required = {"source_path", "start_frame", "end_frame", "new_name"}
    rows: list[IndexRow] = []
    sources: set[str] = set()
    names: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise PipelineError(
                f"index.csv missing columns {sorted(required)}"
            )
        for row_number, raw in enumerate(reader, start=2):
            relative = source_relative(raw["source_path"])
            source = PurePosixPath(relative)
            humanact = source.parts[0] == "humanact12"
            if humanact and (
                len(source.parts) < 3
                or source.parts[:2] != ("humanact12", "humanact12")
            ):
                raise SafetyError(
                    f"index row {row_number}: invalid HumanAct layout"
                )
            start = parse_frame(
                raw["start_frame"],
                "start_frame",
                row_number,
                allow_minus_one=humanact,
            )
            end = parse_frame(
                raw["end_frame"],
                "end_frame",
                row_number,
                allow_minus_one=humanact,
            )
            if humanact:
                if (start, end) not in ((0, -1), (-1, -1)):
                    raise PipelineError(
                        f"index row {row_number}: unexpected HumanAct slice"
                    )
            elif end <= start:
                raise PipelineError(
                    f"index row {row_number}: non-positive segment"
                )
            name = normalize_relative(raw["new_name"])
            if "/" in name or not name.endswith(".npy"):
                raise PipelineError(
                    f"index row {row_number}: invalid new_name={name!r}"
                )
            if name in names or f"M{name}" in names:
                raise PipelineError(
                    f"index row {row_number}: duplicate/colliding name"
                )
            names.add(name)
            sources.add(relative)
            top = source.parts[0]
            rows.append(
                IndexRow(
                    source_rel=relative,
                    start_frame=start,
                    end_frame=end,
                    new_name=name,
                    humanact=humanact,
                    trim_frames=SOURCE_TRIM_FRAMES.get(top, 0),
                )
            )
    mirrored = {f"M{name}" for name in names}
    segment_names = names | mirrored
    represent_names = segment_names - set(EXPECTED_SHORT_IDS)
    if enforce_official_counts:
        gates = (
            (len(rows), OFFICIAL_INDEX_ROWS, "index rows"),
            (len(sources), OFFICIAL_POSE_SOURCES, "pose sources"),
            (
                len(segment_names),
                OFFICIAL_SEGMENT_FILES,
                "segment outputs",
            ),
            (
                len(represent_names),
                OFFICIAL_REPRESENT_FILES,
                "representation outputs",
            ),
        )
        for observed, expected, label in gates:
            if observed != expected:
                raise PipelineError(
                    f"{label}: expected {expected}, found {observed}"
                )
        expected_sequence = [
            f"{index:06d}.npy" for index in range(OFFICIAL_INDEX_ROWS)
        ]
        if [row.new_name for row in rows] != expected_sequence:
            raise PipelineError("index new_name sequence is not canonical")
    return IndexContract(
        rows=tuple(rows),
        unique_sources=frozenset(sources),
        segment_names=frozenset(segment_names),
        represent_names=frozenset(represent_names),
    )


def existing_device(path: Path) -> int:
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise SafetyError(f"cannot resolve device for {path}")
        current = parent
    return current.stat().st_dev


def validate_layout(
    source_root: Path, staging_root: Path, data_root: Path
) -> None:
    data = data_root.resolve()
    source = source_root.resolve()
    staging = staging_root.resolve()
    if not data.is_dir():
        raise SafetyError(f"data root is missing: {data}")
    if data_root.is_symlink():
        raise SafetyError(f"data root is a symlink: {data_root}")
    if not source.is_dir():
        raise SafetyError(f"official source root is missing: {source}")
    if not source.is_relative_to(data):
        raise SafetyError("official source root is outside the data disk")
    if not staging.is_relative_to(data):
        raise SafetyError("staging root is outside the data disk")
    data_lexical = Path(os.path.abspath(data_root))
    staging_lexical = Path(os.path.abspath(staging_root))
    try:
        staging_relative = staging_lexical.relative_to(data_lexical)
    except ValueError as exc:
        raise SafetyError("staging lexical path is outside the data disk") from exc
    current = data_lexical
    for part in staging_relative.parts:
        current = current / part
        if current.is_symlink():
            raise SafetyError(f"symlink in staging root: {current}")
    if staging == source or staging.is_relative_to(source):
        raise SafetyError("staging root overlaps the official source")
    if source.is_relative_to(staging):
        raise SafetyError("official source overlaps the staging root")
    if existing_device(staging_root) != existing_device(data_root):
        raise SafetyError("staging root is not on the data-disk device")


def configure_runtime(staging_root: Path) -> dict[str, Path]:
    paths = {
        "TMPDIR": checked_path(staging_root, "runtime/postprocess/tmp"),
        "XDG_CACHE_HOME": checked_path(
            staging_root, "runtime/postprocess/xdg-cache"
        ),
        "MPLCONFIGDIR": checked_path(
            staging_root, "runtime/postprocess/matplotlib"
        ),
        "PYTHONPYCACHEPREFIX": checked_path(
            staging_root, "runtime/postprocess/pycache"
        ),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return paths


def force_cpu_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


class SpaceGuard:
    def __init__(self, root: Path, reserve_bytes: int):
        if reserve_bytes < 0:
            raise PipelineError("disk reserve must be nonnegative")
        self.root = root
        self.reserve_bytes = reserve_bytes

    def require(self, extra_bytes: int = 0, *, label: str) -> None:
        if extra_bytes < 0:
            raise PipelineError(f"{label}: negative disk requirement")
        free = shutil.disk_usage(self.root).free
        required = self.reserve_bytes + extra_bytes
        if free < required:
            raise PipelineError(
                f"{label}: free={free}, required={required}"
            )


class FileLock:
    def __init__(
        self, path: Path, *, shared: bool = False, blocking: bool = False
    ):
        self.path = path
        self.shared = shared
        self.blocking = blocking
        self.stream = None

    def __enter__(self) -> FileLock:  # noqa: PYI034 - Python 3.10
        reject_symlink_components(self.path)
        flags = os.O_RDONLY if self.shared else os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(
            os, "O_CLOEXEC", 0
        )
        if not self.shared:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise PipelineError(
                f"cannot safely open pipeline lock: {self.path}: {exc}"
            ) from exc
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise PipelineError(f"pipeline lock is not regular: {self.path}")
        self.stream = os.fdopen(
            descriptor,
            "r" if self.shared else "r+",
            encoding="utf-8",
        )
        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(self.stream.fileno(), operation)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise PipelineError(f"pipeline lock is busy: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def validate_joints_array(
    array: object,
    *,
    label: str,
    expected_frames: int | None = None,
) -> None:
    np = import_numpy()
    if not isinstance(array, np.ndarray):
        raise PipelineError(f"{label}: expected ndarray")
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1:] != (JOINTS_NUM, 3)
    ):
        raise PipelineError(f"{label}: invalid joints shape {array.shape}")
    if expected_frames is not None and array.shape[0] != expected_frames:
        raise PipelineError(
            f"{label}: expected {expected_frames} frames, "
            f"found {array.shape[0]}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise PipelineError(f"{label}: non-floating dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise PipelineError(f"{label}: contains NaN or Inf")


def validate_segment_array(
    array: object,
    *,
    label: str,
    expected_frames: int | None = None,
    expected_joints: int | None = None,
) -> None:
    np = import_numpy()
    if not isinstance(array, np.ndarray):
        raise PipelineError(f"{label}: expected ndarray")
    allowed_joints = {JOINTS_NUM, HUMANACT_JOINTS}
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] not in allowed_joints
        or array.shape[2] != 3
    ):
        raise PipelineError(f"{label}: invalid segment shape {array.shape}")
    if expected_frames is not None and array.shape[0] != expected_frames:
        raise PipelineError(
            f"{label}: expected {expected_frames} frames, "
            f"found {array.shape[0]}"
        )
    if expected_joints is not None and array.shape[1] != expected_joints:
        raise PipelineError(
            f"{label}: expected {expected_joints} joints, "
            f"found {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise PipelineError(f"{label}: non-floating dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise PipelineError(f"{label}: contains NaN or Inf")


def validate_vector_array(
    array: object,
    *,
    label: str,
    expected_frames: int | None = None,
) -> None:
    np = import_numpy()
    if not isinstance(array, np.ndarray):
        raise PipelineError(f"{label}: expected ndarray")
    if (
        array.ndim != 2
        or array.shape[0] < 1
        or array.shape[1] != FEATURE_DIM
    ):
        raise PipelineError(f"{label}: invalid vector shape {array.shape}")
    if expected_frames is not None and array.shape[0] != expected_frames:
        raise PipelineError(
            f"{label}: expected {expected_frames} frames, "
            f"found {array.shape[0]}"
        )
    if array.dtype != np.dtype("float32"):
        raise PipelineError(f"{label}: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise PipelineError(f"{label}: contains NaN or Inf")


def validate_stat_array(array: object, *, label: str) -> None:
    np = import_numpy()
    if not isinstance(array, np.ndarray) or array.shape != (FEATURE_DIM,):
        shape = getattr(array, "shape", None)
        raise PipelineError(f"{label}: invalid statistics shape {shape}")
    if array.dtype != np.dtype("float32"):
        raise PipelineError(f"{label}: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise PipelineError(f"{label}: contains NaN or Inf")


def load_npy(
    path: Path,
    validator: Callable[..., None],
    **validator_kwargs: object,
):
    np = import_numpy()
    if not path.is_file() or path.is_symlink():
        raise PipelineError(f"missing or unsafe npy: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"cannot load {path}: {exc}") from exc
    validator(array, label=str(path), **validator_kwargs)
    return array


def atomic_npy(
    path: Path,
    array: object,
    validator: Callable[..., None],
    *,
    space_guard: SpaceGuard | None = None,
    **validator_kwargs: object,
) -> NpyInfo:
    np = import_numpy()
    validator(array, label=str(path), **validator_kwargs)
    if space_guard is not None:
        space_guard.require(
            int(getattr(array, "nbytes", 0)) + 4096,
            label=f"write {path}",
        )
    reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    persisted = load_npy(path, validator, **validator_kwargs)
    info = NpyInfo(
        shape=tuple(int(value) for value in persisted.shape),
        dtype=str(persisted.dtype),
        size=path.stat().st_size,
        sha256=sha256_file(path),
    )
    if space_guard is not None:
        space_guard.require(label=f"post-write {path}")
    return info


class StateDB:
    """A postprocess-only durable resume ledger."""

    def __init__(self, path: Path, staging_root: Path):
        reject_symlink_components(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(path)
        reject_sqlite_symlinks(path)
        before = path.stat() if path.exists() else None
        if before is not None and not stat.S_ISREG(before.st_mode):
            raise SafetyError(f"state database is not regular: {path}")
        self.staging_root = staging_root.resolve()
        uri = f"file:{path.absolute().as_posix()}?mode=rwc&nofollow=1"
        self.db = sqlite3.connect(uri, uri=True)
        after = path.stat()
        if (
            not stat.S_ISREG(after.st_mode)
            or (
                before is not None
                and (before.st_dev, before.st_ino)
                != (after.st_dev, after.st_ino)
            )
        ):
            self.db.close()
            raise SafetyError(
                f"state database identity changed while opening: {path}"
            )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS items(
                stage TEXT NOT NULL,
                item_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                output_rel TEXT NOT NULL,
                output_size INTEGER NOT NULL,
                output_sha256 TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(stage,item_key))
            """
        )
        self.db.commit()

    def __enter__(self) -> StateDB:  # noqa: PYI034 - Python 3.10
        return self

    def __exit__(self, *_: object) -> None:
        self.db.close()

    def complete(
        self,
        stage: str,
        key: str,
        fingerprint: str,
        output_rel: str,
        validator: Callable[..., None],
        *,
        verify_hash: bool,
        validator_kwargs: Mapping[str, object] | None = None,
    ) -> bool:
        row = self.db.execute(
            "SELECT fingerprint,output_rel,output_size,output_sha256 "
            "FROM items WHERE stage=? AND item_key=?",
            (stage, key),
        ).fetchone()
        if row is None or row[0] != fingerprint or row[1] != output_rel:
            return False
        try:
            output = checked_path(self.staging_root, output_rel)
            if not output.is_file() or output.stat().st_size != row[2]:
                return False
            load_npy(output, validator, **dict(validator_kwargs or {}))
            return not verify_hash or sha256_file(output) == row[3]
        except PipelineError:
            return False

    def record(
        self,
        stage: str,
        key: str,
        fingerprint: str,
        output_rel: str,
        info: NpyInfo,
    ) -> None:
        normalized = normalize_relative(output_rel)
        self.db.execute(
            """
            INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(stage,item_key) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                output_rel=excluded.output_rel,
                output_size=excluded.output_size,
                output_sha256=excluded.output_sha256,
                shape_json=excluded.shape_json,
                dtype=excluded.dtype,
                updated_at=excluded.updated_at
            """,
            (
                stage,
                key,
                fingerprint,
                normalized,
                info.size,
                info.sha256,
                json.dumps(list(info.shape)),
                info.dtype,
                utc_now(),
            ),
        )
        self.db.commit()

    def records(self, stage: str) -> dict[str, dict[str, object]]:
        rows = self.db.execute(
            "SELECT item_key,fingerprint,output_rel,output_size,"
            "output_sha256,shape_json,dtype FROM items WHERE stage=?",
            (stage,),
        ).fetchall()
        return {
            row[0]: {
                "fingerprint": row[1],
                "output_rel": row[2],
                "output_size": row[3],
                "output_sha256": row[4],
                "shape": json.loads(row[5]),
                "dtype": row[6],
            }
            for row in rows
        }

    def counts(self) -> dict[str, int]:
        return dict(
            self.db.execute(
                "SELECT stage,COUNT(*) FROM items GROUP BY stage"
            ).fetchall()
        )


def reject_sqlite_symlinks(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.is_symlink():
            raise SafetyError(f"SQLite path is a symlink: {candidate}")


def read_state_counts(path: Path) -> dict[str, int]:
    """Read an existing resume ledger without initializing or migrating it."""
    reject_sqlite_symlinks(path)
    uri = f"file:{path.absolute().as_posix()}?mode=ro&nofollow=1"
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(uri, uri=True)
        database.execute("PRAGMA query_only=ON")
        return dict(
            database.execute(
                "SELECT stage,COUNT(*) FROM items GROUP BY stage"
            ).fetchall()
        )
    except sqlite3.Error as exc:
        raise PipelineError(
            f"cannot read postprocess state database {path}: {exc}"
        ) from exc
    finally:
        if database is not None:
            database.close()


def load_pose_ledger(
    staging_root: Path,
    contract: IndexContract,
    pose_success: Mapping[str, object],
) -> Mapping[str, str]:
    path = checked_path(staging_root, "state.sqlite3")
    reject_sqlite_symlinks(path)
    uri = f"file:{path.absolute().as_posix()}?mode=ro&nofollow=1"
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(uri, uri=True)
        database.execute("PRAGMA query_only=ON")
        rows = database.execute(
            "SELECT item_key,fingerprint,output_rel,output_size,"
            "output_sha256,shape_json,dtype FROM items WHERE stage='pose'"
        ).fetchall()
    except sqlite3.Error as exc:
        raise PipelineError(
            f"cannot read pose state database {path}: {exc}"
        ) from exc
    finally:
        if database is not None:
            database.close()
    expected = set(contract.unique_sources)
    observed = {str(row[0]) for row in rows}
    if len(observed) != len(rows) or observed != expected:
        raise PipelineError(
            "pose ledger source set mismatch: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )
    pose_fingerprint = str(pose_success["pose_fingerprint"])
    hashes: dict[str, str] = {}
    ledger: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda value: str(value[0])):
        key = str(row[0])
        fingerprint = str(row[1])
        output_rel = str(row[2])
        digest = validate_sha256(
            row[4], label=f"pose ledger {key}"
        )
        try:
            shape = json.loads(str(row[5]))
        except ValueError as exc:
            raise PipelineError(
                f"pose ledger {key}: invalid shape"
            ) from exc
        if (
            output_rel != key
            or not fingerprint.startswith(f"{pose_fingerprint}:")
            or int(row[3]) <= 0
            or not isinstance(shape, list)
            or len(shape) != 3
            or int(shape[0]) < 1
            or int(shape[1]) < JOINTS_NUM
            or int(shape[2]) != 3
            or str(row[6]) not in ("float32", "float64")
        ):
            raise PipelineError(f"pose ledger binding is invalid: {key}")
        hashes[key] = digest
        ledger.append(
            {
                "source": key,
                "fingerprint": fingerprint,
                "output_sha256": digest,
            }
        )
    if canonical_sha256(ledger) != pose_success["record_set_sha256"]:
        raise PipelineError("pose ledger record set does not match marker")
    return hashes


def load_json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PipelineError(f"cannot load {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PipelineError(f"{label} is not a JSON object")
    return value


def load_pose_success(
    staging_root: Path,
    index_sha256: str,
    *,
    enforce_official_counts: bool = True,
) -> tuple[Mapping[str, object], str]:
    path = manifest_path(staging_root, "pose-success.json")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PipelineError(
            f"cannot load pose success marker: {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise PipelineError("pose success marker is not a JSON object")
    if value.get("schema_version") != POSE_SCHEMA_VERSION:
        raise PipelineError("pose success marker schema mismatch")
    if value.get("index_sha256") != index_sha256:
        raise PipelineError("pose success marker index hash mismatch")
    if value.get("finite") is not True:
        raise PipelineError("pose success marker is not finite")
    if int(value.get("joint_floor", 0)) < JOINTS_NUM:
        raise PipelineError("pose success marker has too few joints")
    sources = int(value.get("sources", -1))
    if enforce_official_counts and sources != OFFICIAL_POSE_SOURCES:
        raise PipelineError(
            f"pose marker expected {OFFICIAL_POSE_SOURCES} sources, "
            f"found {sources}"
        )
    validate_sha256(
        value.get("pose_fingerprint"), label="pose fingerprint"
    )
    validate_sha256(
        value.get("record_set_sha256"), label="pose record set"
    )
    return value, hashlib.sha256(raw).hexdigest()


def algorithm_contract(source_root: Path) -> dict[str, object]:
    inputs = {
        relative: sha256_file(source_root / relative)
        for relative in (
            "paramUtil.py",
            "common/skeleton.py",
            "common/quaternion.py",
        )
    }
    payload: dict[str, object] = {
        "revision": ALGORITHM_REVISION,
        "script_sha256": FROZEN_UPSTREAM_SCRIPT_SHA256,
        "official_common_sha256": inputs,
        "joints_num": JOINTS_NUM,
        "humanact_segment_joints": HUMANACT_JOINTS,
        "feature_dim": FEATURE_DIM,
        "repair_epsilon": REPAIR_EPSILON,
        "repaired_nonfinite_ids": sorted(REPAIRED_NONFINITE_IDS),
        "expected_short_ids": sorted(EXPECTED_SHORT_IDS),
        "official_reference_sha256": dict(REFERENCE_HASHES),
    }
    upstream = {**payload, "fingerprint": canonical_sha256(payload)}
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    stats_payload: dict[str, object] = {
        "revision": STATISTICS_ALGORITHM_REVISION,
        "script_sha256": implementation_sha256,
        "official_compatibility": {
            "reduction": "np.concatenate float32 population mean/std",
            "order": "os.listdir",
            "mean_reference_atol": LEGACY_MEAN_REFERENCE_ATOL,
            "std_reference_atol": LEGACY_STD_REFERENCE_ATOL,
            "ram_multiplier": LEGACY_RAM_MULTIPLIER,
            "ram_safety_bytes": LEGACY_RAM_SAFETY_BYTES,
        },
        "stable": {
            "reduction": "batch-merged Welford population moments",
            "accumulator_dtype": "float64",
            "output_dtype": "float32",
            "ddof": 0,
        },
    }
    return {
        **upstream,
        "implementation_sha256": implementation_sha256,
        "statistics": {
            **stats_payload,
            "fingerprint": canonical_sha256(stats_payload),
        },
    }


def stage_fingerprint(
    stage: str,
    algorithm: Mapping[str, object],
    index_sha256: str,
    input_fingerprint: str,
) -> str:
    return canonical_sha256(
        {
            "stage": stage,
            "algorithm": algorithm["fingerprint"],
            "index_sha256": index_sha256,
            "input_fingerprint": input_fingerprint,
        }
    )


def load_stage_success(
    path: Path, stage: str, expected_fingerprint: str
) -> Mapping[str, object]:
    value = load_json_mapping(path, label=f"{stage} success marker")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError(f"{path}: schema mismatch")
    if value.get("stage") != stage:
        raise PipelineError(f"{path}: wrong stage marker")
    if value.get("fingerprint") != expected_fingerprint:
        raise PipelineError(f"{stage} success fingerprint mismatch")
    if value.get("finite") is not True:
        raise PipelineError(f"{stage} success is not finite")
    return value


def swap_left_right(data):
    validate_segment_array(data, label="mirror source")
    mirrored = data.copy()
    mirrored[..., 0] *= -1
    temporary = mirrored[:, RIGHT_CHAIN].copy()
    mirrored[:, RIGHT_CHAIN] = mirrored[:, LEFT_CHAIN]
    mirrored[:, LEFT_CHAIN] = temporary
    validate_segment_array(
        mirrored,
        label="mirrored joints",
        expected_frames=data.shape[0],
        expected_joints=data.shape[1],
    )
    return mirrored


def segment_positions(source, row: IndexRow):
    validate_segment_array(
        source,
        label=row.source_rel,
        expected_joints=row.segment_joints,
    )
    if row.humanact:
        data = source.copy()
    else:
        required = row.trim_frames + row.end_frame
        if source.shape[0] < required:
            raise PipelineError(
                f"{row.source_rel}: needs {required} frames after contract, "
                f"found {source.shape[0]}"
            )
        trimmed = source[row.trim_frames :]
        data = trimmed[row.start_frame : row.end_frame].copy()
        data[..., 0] *= -1
    validate_segment_array(
        data,
        label=row.new_name,
        expected_frames=row.output_frames,
        expected_joints=row.segment_joints,
    )
    mirrored = swap_left_right(data)
    return data, mirrored


def list_relative_files(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    unsafe: list[str] = []
    if not root.is_dir() or root.is_symlink():
        raise PipelineError(f"missing or unsafe output directory: {root}")
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            unsafe.append(relative)
        elif candidate.is_file():
            files.add(relative)
    return files, unsafe


def verify_segment_tree(
    contract: IndexContract,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    *,
    verify_hash: bool,
    enforce_official_counts: bool,
) -> dict[str, object]:
    np = import_numpy()
    joints_root = checked_path(staging_root, "joints")
    observed, unsafe = list_relative_files(joints_root)
    expected = set(contract.segment_names)
    if unsafe or observed != expected:
        raise PipelineError(
            "segment file set mismatch: "
            f"unsafe={unsafe[:5]}, "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )
    records = state.records("segment")
    expected_keys = {f"joints/{name}" for name in expected}
    if set(records) != expected_keys:
        raise PipelineError(
            "segment ledger mismatch: "
            f"missing={sorted(expected_keys - set(records))[:5]}, "
            f"extra={sorted(set(records) - expected_keys)[:5]}"
        )
    ledger: list[dict[str, object]] = []
    frames = 0
    one_frame: set[str] = set()
    for row in contract.rows:
        base_path = checked_path(joints_root, row.new_name)
        mirror_name = f"M{row.new_name}"
        mirror_path = checked_path(joints_root, mirror_name)
        base = load_npy(
            base_path,
            validate_segment_array,
            expected_frames=row.output_frames,
            expected_joints=row.segment_joints,
        )
        mirror = load_npy(
            mirror_path,
            validate_segment_array,
            expected_frames=base.shape[0],
            expected_joints=row.segment_joints,
        )
        if not np.array_equal(swap_left_right(base), mirror):
            raise PipelineError(
                f"{mirror_name}: does not exactly mirror {row.new_name}"
            )
        frames += int(base.shape[0] + mirror.shape[0])
        if base.shape[0] == 1:
            one_frame.update((row.new_name, mirror_name))
        for name, path in (
            (row.new_name, base_path),
            (mirror_name, mirror_path),
        ):
            output_rel = f"joints/{name}"
            record = records[output_rel]
            if (
                record["output_rel"] != output_rel
                or not str(record["fingerprint"]).startswith(
                    f"{fingerprint}:"
                )
                or path.stat().st_size != record["output_size"]
            ):
                raise PipelineError(
                    f"{output_rel}: resume ledger binding mismatch"
                )
            digest = str(record["output_sha256"])
            if verify_hash and sha256_file(path) != digest:
                raise PipelineError(f"{output_rel}: digest mismatch")
            ledger.append({"output": output_rel, "sha256": digest})
    if enforce_official_counts:
        if len(expected) != OFFICIAL_SEGMENT_FILES:
            raise PipelineError("segment count gate failed")
        if one_frame != set(EXPECTED_SHORT_IDS):
            raise PipelineError(
                f"expected one-frame set changed: {sorted(one_frame)}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "segment",
        "verified_at": utc_now(),
        "fingerprint": fingerprint,
        "files": len(expected),
        "frames": frames,
        "finite": True,
        "mirror_verified": True,
        "record_set_sha256": canonical_sha256(
            sorted(ledger, key=lambda item: str(item["output"]))
        ),
        "expected_short_ids": sorted(EXPECTED_SHORT_IDS),
    }


def segment_dataset(
    contract: IndexContract,
    pose_root: Path,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    pose_hashes: Mapping[str, str],
    space_guard: SpaceGuard,
    *,
    verify_resume_hash: bool,
    enforce_official_counts: bool,
) -> dict[str, object]:
    joints_root = checked_path(staging_root, "joints")
    joints_root.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temps(joints_root, set(contract.segment_names))
    source_hashes: dict[str, str] = {}
    completed = 0
    for row in contract.rows:
        source_path = checked_path(pose_root, row.source_rel)
        if row.source_rel not in source_hashes:
            source_hashes[row.source_rel] = sha256_file(source_path)
            if (
                source_hashes[row.source_rel]
                != pose_hashes.get(row.source_rel)
            ):
                raise PipelineError(
                    f"pose source hash changed: {row.source_rel}"
                )
        row_fingerprint = canonical_sha256(
            {
                "stage": fingerprint,
                "source": row.source_rel,
                "source_sha256": source_hashes[row.source_rel],
                "start": row.start_frame,
                "end": row.end_frame,
                "trim": row.trim_frames,
                "humanact": row.humanact,
            }
        )
        base_rel = f"joints/{row.new_name}"
        mirror_name = f"M{row.new_name}"
        mirror_rel = f"joints/{mirror_name}"
        base_complete = state.complete(
            "segment",
            base_rel,
            f"{fingerprint}:{row_fingerprint}:base",
            base_rel,
            validate_segment_array,
            verify_hash=verify_resume_hash,
            validator_kwargs={
                "expected_frames": row.output_frames,
                "expected_joints": row.segment_joints,
            },
        )
        mirror_complete = state.complete(
            "segment",
            mirror_rel,
            f"{fingerprint}:{row_fingerprint}:mirror",
            mirror_rel,
            validate_segment_array,
            verify_hash=verify_resume_hash,
            validator_kwargs={
                "expected_frames": row.output_frames,
                "expected_joints": row.segment_joints,
            },
        )
        if base_complete and mirror_complete:
            completed += 2
            continue
        source = load_npy(
            source_path,
            validate_segment_array,
            expected_joints=row.segment_joints,
        )
        if sha256_file(source_path) != source_hashes[row.source_rel]:
            raise PipelineError(
                f"pose input changed while segmenting: {row.source_rel}"
            )
        base, mirror = segment_positions(source, row)
        for name, output_rel, array, suffix in (
            (row.new_name, base_rel, base, "base"),
            (mirror_name, mirror_rel, mirror, "mirror"),
        ):
            output = checked_path(staging_root, output_rel)
            info = atomic_npy(
                output,
                array,
                validate_segment_array,
                space_guard=space_guard,
                expected_frames=row.output_frames,
                expected_joints=row.segment_joints,
            )
            state.record(
                "segment",
                output_rel,
                f"{fingerprint}:{row_fingerprint}:{suffix}",
                output_rel,
                info,
            )
            completed += 1
        if completed % 200 == 0:
            print(
                f"SEGMENT {completed}/{len(contract.segment_names)}",
                flush=True,
            )
    result = verify_segment_tree(
        contract,
        staging_root,
        state,
        fingerprint,
        verify_hash=True,
        enforce_official_counts=enforce_official_counts,
    )
    atomic_json(
        manifest_path(
            staging_root, "postprocess-segment-success.json"
        ),
        result,
    )
    return result


def load_official_backend(source_root: Path) -> SimpleNamespace:
    """Import the official common modules from an explicit trusted path."""
    force_cpu_environment()
    np = import_numpy()
    source = source_root.resolve()
    expected = {
        "common.quaternion": (source / "common/quaternion.py").resolve(),
        "common.skeleton": (source / "common/skeleton.py").resolve(),
        "paramUtil": (source / "paramUtil.py").resolve(),
    }
    for name, expected_path in expected.items():
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != expected_path:
            raise SafetyError(
                f"refusing preloaded module {name} from {loaded_file}"
            )
    inserted = False
    source_text = str(source)
    if not sys.path or sys.path[0] != source_text:
        sys.path.insert(0, source_text)
        inserted = True
    added_np_float = False
    if not hasattr(np, "float"):
        np.float = float
        added_np_float = True
    try:
        quaternion = importlib.import_module("common.quaternion")
        skeleton = importlib.import_module("common.skeleton")
        param_util = importlib.import_module("paramUtil")
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(source_text)
        if added_np_float:
            delattr(np, "float")
    for name, module in (
        ("common.quaternion", quaternion),
        ("common.skeleton", skeleton),
        ("paramUtil", param_util),
    ):
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if module_file != expected[name]:
            raise SafetyError(
                f"{name} resolved to {module_file}, expected {expected[name]}"
            )
    torch = skeleton.torch
    n_raw_offsets = torch.from_numpy(param_util.t2m_raw_offsets)
    return SimpleNamespace(
        np=np,
        torch=torch,
        Skeleton=skeleton.Skeleton,
        skeleton_module=skeleton,
        n_raw_offsets=n_raw_offsets,
        raw_offsets=np.asarray(param_util.t2m_raw_offsets),
        kinematic_chain=param_util.t2m_kinematic_chain,
        qbetween_np=quaternion.qbetween_np,
        qrot_np=quaternion.qrot_np,
        qmul_np=quaternion.qmul_np,
        qinv_np=quaternion.qinv_np,
        quaternion_to_cont6d_np=quaternion.quaternion_to_cont6d_np,
        qrot=quaternion.qrot,
        qinv=quaternion.qinv,
    )


def safe_unit_rows(
    values,
    fallback,
    epsilon: float,
    diagnostics: dict[str, int],
):
    np = import_numpy()
    array = np.asarray(values, dtype=np.float64)
    original_shape = array.shape
    if not original_shape or original_shape[-1] != 3:
        raise PipelineError(f"cannot normalize shape {original_shape}")
    rows = array.reshape(-1, 3)
    result = np.empty_like(rows)
    fallback_row = np.asarray(fallback, dtype=np.float64).reshape(3)
    fallback_norm = float(np.linalg.norm(fallback_row))
    if not math.isfinite(fallback_norm) or fallback_norm <= epsilon:
        raise PipelineError("invalid deterministic orientation fallback")
    previous = fallback_row / fallback_norm
    fallback_count = 0
    for index, row in enumerate(rows):
        norm = float(np.linalg.norm(row))
        if math.isfinite(norm) and norm > epsilon:
            previous = row / norm
        else:
            fallback_count += 1
        result[index] = previous
    diagnostics["unit_fallbacks"] += fallback_count
    return result.reshape(original_shape)


def safe_qbetween(
    first,
    second,
    epsilon: float,
    diagnostics: dict[str, int],
):
    np = import_numpy()
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.shape != second_array.shape
        or first_array.shape[-1] != 3
    ):
        raise PipelineError("safe qbetween shape mismatch")
    a = safe_unit_rows(
        first_array, [1.0, 0.0, 0.0], epsilon, diagnostics
    ).reshape(-1, 3)
    b = safe_unit_rows(
        second_array, [1.0, 0.0, 0.0], epsilon, diagnostics
    ).reshape(-1, 3)
    cross = np.cross(a, b)
    scalar = 1.0 + np.sum(a * b, axis=1, keepdims=True)
    quaternion = np.concatenate([scalar, cross], axis=1)
    norms = np.linalg.norm(quaternion, axis=1)
    antiparallel = (~np.isfinite(norms)) | (norms <= epsilon)
    for index in np.flatnonzero(antiparallel):
        vector = a[index]
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(vector)))] = 1.0
        axis = np.cross(vector, basis)
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(axis_norm) or axis_norm <= epsilon:
            raise PipelineError("cannot construct antiparallel fallback")
        quaternion[index] = np.concatenate(
            [[0.0], axis / axis_norm]
        )
    diagnostics["antiparallel_fallbacks"] += int(
        np.count_nonzero(antiparallel)
    )
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / norms
    return quaternion.reshape(first_array.shape[:-1] + (4,)).astype(
        np.float32
    )


def make_safe_skeleton(
    backend: SimpleNamespace,
    epsilon: float,
    diagnostics: dict[str, int],
):
    np = import_numpy()
    base = backend.Skeleton

    class SafeSkeleton(base):
        def inverse_kinematics_np(
            self, joints, face_joint_idx, smooth_forward=False
        ):
            if len(face_joint_idx) != 4:
                raise PipelineError("face_joint_idx must have four joints")
            # Keep the official historical unpacking order for parity.
            l_hip, r_hip, shoulder_r, shoulder_l = face_joint_idx
            across = (
                joints[:, r_hip]
                - joints[:, l_hip]
                + joints[:, shoulder_r]
                - joints[:, shoulder_l]
            )
            across = safe_unit_rows(
                across, [1.0, 0.0, 0.0], epsilon, diagnostics
            )
            forward = np.cross(
                np.array([[0.0, 1.0, 0.0]]), across, axis=-1
            )
            if smooth_forward:
                forward = (
                    backend.skeleton_module.filters.gaussian_filter1d(
                        forward, 20, axis=0, mode="nearest"
                    )
                )
            forward = safe_unit_rows(
                forward, [0.0, 0.0, 1.0], epsilon, diagnostics
            )
            target = np.array([[0.0, 0.0, 1.0]]).repeat(
                len(forward), axis=0
            )
            root_quat = safe_qbetween(
                forward, target, epsilon, diagnostics
            )
            quat_params = np.zeros(joints.shape[:-1] + (4,))
            root_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
            quat_params[:, 0] = root_quat
            for chain in self._kinematic_tree:
                rotation = root_quat
                for chain_index in range(len(chain) - 1):
                    child = chain[chain_index + 1]
                    parent = chain[chain_index]
                    raw = self._raw_offset_np[child]
                    unit = np.repeat(
                        raw[np.newaxis, ...], len(joints), axis=0
                    )
                    direction = joints[:, child] - joints[:, parent]
                    direction = safe_unit_rows(
                        direction, raw, epsilon, diagnostics
                    )
                    raw_to_direction = safe_qbetween(
                        unit, direction, epsilon, diagnostics
                    )
                    local = backend.qmul_np(
                        backend.qinv_np(rotation), raw_to_direction
                    )
                    quat_params[:, child, :] = local
                    rotation = backend.qmul_np(rotation, local)
            if not np.isfinite(quat_params).all():
                raise PipelineError("safe IK produced non-finite rotations")
            return quat_params

    return SafeSkeleton


def target_offsets_from_file(
    target_path: Path, backend: SimpleNamespace
):
    target = load_npy(target_path, validate_segment_array)
    target = target[:, :JOINTS_NUM]
    skeleton = backend.Skeleton(
        backend.n_raw_offsets, backend.kinematic_chain, "cpu"
    )
    first_frame = backend.torch.from_numpy(target[0])
    offsets = skeleton.get_offsets_joints(first_frame)
    if (
        tuple(offsets.shape) != (JOINTS_NUM, 3)
        or not backend.torch.isfinite(offsets).all()
    ):
        raise PipelineError("target skeleton offsets are invalid")
    return offsets


def uniform_skeleton(
    positions,
    target_offsets,
    backend: SimpleNamespace,
    skeleton_class,
    *,
    repair: bool,
    epsilon: float,
):
    np = import_numpy()
    source_skeleton = skeleton_class(
        backend.n_raw_offsets, backend.kinematic_chain, "cpu"
    )
    source_offsets = (
        source_skeleton.get_offsets_joints(
            backend.torch.from_numpy(positions[0])
        )
        .numpy()
    )
    target_array = target_offsets.numpy()
    source_leg = np.abs(source_offsets[L_IDX1]).max() + np.abs(
        source_offsets[L_IDX2]
    ).max()
    target_leg = np.abs(target_array[L_IDX1]).max() + np.abs(
        target_array[L_IDX2]
    ).max()
    if (
        not math.isfinite(float(source_leg))
        or not math.isfinite(float(target_leg))
        or float(target_leg) <= epsilon
    ):
        raise PipelineError("invalid leg length for skeleton normalization")
    if repair and float(source_leg) <= epsilon:
        scale = 1.0
    elif float(source_leg) <= 0:
        raise PipelineError("zero source leg length")
    else:
        scale = float(target_leg / source_leg)
    root_position = positions[:, 0] * scale
    rotations = source_skeleton.inverse_kinematics_np(
        positions, FACE_JOINT_INDEX
    )
    source_skeleton.set_offset(target_offsets)
    output = source_skeleton.forward_kinematics_np(
        rotations, root_position
    )
    if not np.isfinite(output).all():
        raise PipelineError("uniform skeleton produced NaN or Inf")
    return output


def recover_from_ric(data, backend: SimpleNamespace):
    torch = backend.torch
    tensor = torch.from_numpy(data).unsqueeze(0).float()
    rotation_velocity = tensor[..., 0]
    rotation_angle = torch.zeros_like(rotation_velocity)
    rotation_angle[..., 1:] = rotation_velocity[..., :-1]
    rotation_angle = torch.cumsum(rotation_angle, dim=-1)

    rotation_quaternion = torch.zeros(
        tensor.shape[:-1] + (4,), dtype=tensor.dtype
    )
    rotation_quaternion[..., 0] = torch.cos(rotation_angle)
    rotation_quaternion[..., 2] = torch.sin(rotation_angle)

    root_position = torch.zeros(
        tensor.shape[:-1] + (3,), dtype=tensor.dtype
    )
    root_position[..., 1:, [0, 2]] = tensor[..., :-1, 1:3]
    root_position = backend.qrot(
        backend.qinv(rotation_quaternion), root_position
    )
    root_position = torch.cumsum(root_position, dim=-2)
    root_position[..., 1] = tensor[..., 3]

    positions = tensor[..., 4 : (JOINTS_NUM - 1) * 3 + 4]
    positions = positions.view(
        positions.shape[:-1] + (-1, 3)
    )
    rotations = backend.qinv(
        rotation_quaternion[..., None, :]
    ).expand(positions.shape[:-1] + (4,))
    positions = backend.qrot(rotations, positions)
    positions[..., 0] += root_position[..., 0:1]
    positions[..., 2] += root_position[..., 2:3]
    positions = torch.cat(
        [root_position.unsqueeze(-2), positions], dim=-2
    )
    return positions.squeeze(0).numpy()


def process_motion(
    source,
    target_offsets,
    backend: SimpleNamespace,
    *,
    repair: bool = False,
    epsilon: float = REPAIR_EPSILON,
):
    np = import_numpy()
    validate_joints_array(source, label="representation source")
    if source.shape[0] < 2:
        raise PipelineError("representation source has fewer than 2 frames")
    diagnostics = {
        "unit_fallbacks": 0,
        "antiparallel_fallbacks": 0,
        "arcsin_clips": 0,
    }
    skeleton_class = (
        make_safe_skeleton(backend, epsilon, diagnostics)
        if repair
        else backend.Skeleton
    )
    positions = uniform_skeleton(
        np.asarray(source).copy(),
        target_offsets,
        backend,
        skeleton_class,
        repair=repair,
        epsilon=epsilon,
    )

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    root_position_initial = positions[0].copy()
    root_xz = root_position_initial[0] * np.array([1.0, 0.0, 1.0])
    positions = positions - root_xz

    right_hip, left_hip, shoulder_r, shoulder_l = FACE_JOINT_INDEX
    across = (
        root_position_initial[right_hip]
        - root_position_initial[left_hip]
        + root_position_initial[shoulder_r]
        - root_position_initial[shoulder_l]
    )[None]
    if repair:
        across = safe_unit_rows(
            across, [1.0, 0.0, 0.0], epsilon, diagnostics
        )
    else:
        across = across / np.sqrt((across**2).sum(axis=-1))[
            ..., np.newaxis
        ]
    forward = np.cross(
        np.array([[0.0, 1.0, 0.0]]), across, axis=-1
    )
    if repair:
        forward = safe_unit_rows(
            forward, [0.0, 0.0, 1.0], epsilon, diagnostics
        )
        root_quaternion_initial = safe_qbetween(
            forward,
            np.array([[0.0, 0.0, 1.0]]),
            epsilon,
            diagnostics,
        )
    else:
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[
            ..., np.newaxis
        ]
        root_quaternion_initial = backend.qbetween_np(
            forward, np.array([[0.0, 0.0, 1.0]])
        )
    root_quaternion_initial = (
        np.ones(positions.shape[:-1] + (4,))
        * root_quaternion_initial
    )
    positions = backend.qrot_np(root_quaternion_initial, positions)
    global_positions = positions.copy()

    left_delta = positions[1:, FID_L] - positions[:-1, FID_L]
    right_delta = positions[1:, FID_R] - positions[:-1, FID_R]
    feet_left = (
        (left_delta**2).sum(axis=-1) < FEET_THRESHOLD
    ).astype(np.float32)
    feet_right = (
        (right_delta**2).sum(axis=-1) < FEET_THRESHOLD
    ).astype(np.float32)

    skeleton = skeleton_class(
        backend.n_raw_offsets, backend.kinematic_chain, "cpu"
    )
    quaternion_parameters = skeleton.inverse_kinematics_np(
        positions, FACE_JOINT_INDEX, smooth_forward=True
    )
    continuous_6d = backend.quaternion_to_cont6d_np(
        quaternion_parameters
    )
    root_rotation = quaternion_parameters[:, 0].copy()
    velocity = positions[1:, 0] - positions[:-1, 0]
    velocity = backend.qrot_np(root_rotation[1:], velocity)
    root_velocity_quaternion = backend.qmul_np(
        root_rotation[1:], backend.qinv_np(root_rotation[:-1])
    )

    rifke = positions.copy()
    rifke[..., 0] -= rifke[:, 0:1, 0]
    rifke[..., 2] -= rifke[:, 0:1, 2]
    rifke = backend.qrot_np(
        np.repeat(root_rotation[:, None], rifke.shape[1], axis=1),
        rifke,
    )

    root_y = rifke[:, 0, 1:2]
    yaw_component = root_velocity_quaternion[:, 2:3]
    if repair:
        clipped = np.clip(yaw_component, -1.0, 1.0)
        diagnostics["arcsin_clips"] += int(
            np.count_nonzero(clipped != yaw_component)
        )
        yaw_component = clipped
    angular_velocity = np.arcsin(yaw_component)
    linear_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate(
        [angular_velocity, linear_velocity, root_y[:-1]], axis=-1
    )

    rotation_data = continuous_6d[:, 1:].reshape(
        len(continuous_6d), -1
    )
    ric_data = rifke[:, 1:].reshape(len(rifke), -1)
    local_velocity = backend.qrot_np(
        np.repeat(
            root_rotation[:-1, None],
            global_positions.shape[1],
            axis=1,
        ),
        global_positions[1:] - global_positions[:-1],
    ).reshape(len(global_positions) - 1, -1)
    data = np.concatenate(
        [
            root_data,
            ric_data[:-1],
            rotation_data[:-1],
            local_velocity,
            feet_left,
            feet_right,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    if data.shape != (source.shape[0] - 1, FEATURE_DIM):
        raise PipelineError(f"unexpected representation shape {data.shape}")
    recovered = recover_from_ric(data, backend).astype(
        np.float32, copy=False
    )
    validate_vector_array(
        data,
        label="new_joint_vecs",
        expected_frames=source.shape[0] - 1,
    )
    validate_joints_array(
        recovered,
        label="new_joints",
        expected_frames=source.shape[0] - 1,
    )
    if repair and not any(diagnostics.values()):
        diagnostics["deterministic_repair_path"] = 1
    return recovered, data, diagnostics


def compare_reference_array(
    generated,
    reference_path: Path,
    validator: Callable[..., None],
    *,
    rtol: float,
    atol: float,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    np = import_numpy()
    reference = load_npy(reference_path, validator)
    reference_sha256 = sha256_file(reference_path)
    if (
        expected_sha256 is not None
        and reference_sha256 != expected_sha256
    ):
        raise PipelineError(
            f"official reference hash mismatch for {reference_path}"
        )
    if reference.shape != generated.shape:
        raise PipelineError(
            f"reference shape mismatch: {reference.shape} vs "
            f"{generated.shape}"
        )
    difference = np.abs(
        generated.astype(np.float64) - reference.astype(np.float64)
    )
    maximum = float(difference.max()) if difference.size else 0.0
    mean = float(difference.mean()) if difference.size else 0.0
    close = bool(np.allclose(generated, reference, rtol=rtol, atol=atol))
    if not close:
        raise PipelineError(
            f"reference parity failed for {reference_path}: "
            f"max_abs={maximum}, mean_abs={mean}"
        )
    return {
        "reference": str(reference_path),
        "reference_sha256": reference_sha256,
        "max_abs": maximum,
        "mean_abs": mean,
        "rtol": rtol,
        "atol": atol,
        "allclose": True,
    }


def verify_representation_tree(
    contract: IndexContract,
    source_root: Path,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    repair_details: Mapping[str, Mapping[str, int]],
    *,
    verify_hash: bool,
    enforce_official_counts: bool,
    compare_reference: bool,
) -> dict[str, object]:
    joints_root = checked_path(staging_root, "HumanML3D/new_joints")
    vectors_root = checked_path(
        staging_root, "HumanML3D/new_joint_vecs"
    )
    expected = set(contract.represent_names)
    for root, label in (
        (joints_root, "new_joints"),
        (vectors_root, "new_joint_vecs"),
    ):
        observed, unsafe = list_relative_files(root)
        if unsafe or observed != expected:
            raise PipelineError(
                f"{label} file set mismatch: unsafe={unsafe[:5]}, "
                f"missing={sorted(expected - observed)[:5]}, "
                f"extra={sorted(observed - expected)[:5]}"
            )
    for short_name in EXPECTED_SHORT_IDS & contract.segment_names:
        if (joints_root / short_name).exists() or (
            vectors_root / short_name
        ).exists():
            raise PipelineError(
                f"one-frame expected skip was materialized: {short_name}"
            )
    records = state.records("represent")
    expected_keys = {
        f"HumanML3D/{kind}/{name}"
        for kind in ("new_joints", "new_joint_vecs")
        for name in expected
    }
    if set(records) != expected_keys:
        raise PipelineError(
            "representation ledger mismatch: "
            f"missing={sorted(expected_keys - set(records))[:5]}, "
            f"extra={sorted(set(records) - expected_keys)[:5]}"
        )
    ledger: list[dict[str, object]] = []
    total_frames = 0
    for name in sorted(expected):
        joints_path = checked_path(joints_root, name)
        vectors_path = checked_path(vectors_root, name)
        vectors = load_npy(vectors_path, validate_vector_array)
        load_npy(
            joints_path,
            validate_joints_array,
            expected_frames=vectors.shape[0],
        )
        total_frames += int(vectors.shape[0])
        for kind, path in (
            ("new_joints", joints_path),
            ("new_joint_vecs", vectors_path),
        ):
            output_rel = f"HumanML3D/{kind}/{name}"
            record = records[output_rel]
            if (
                record["output_rel"] != output_rel
                or not str(record["fingerprint"]).startswith(
                    f"{fingerprint}:"
                )
                or path.stat().st_size != record["output_size"]
            ):
                raise PipelineError(
                    f"{output_rel}: resume ledger binding mismatch"
                )
            digest = str(record["output_sha256"])
            if verify_hash and sha256_file(path) != digest:
                raise PipelineError(f"{output_rel}: digest mismatch")
            ledger.append({"output": output_rel, "sha256": digest})
    repaired = set(repair_details)
    expected_repairs = set(REPAIRED_NONFINITE_IDS) & expected
    if repaired != expected_repairs:
        raise PipelineError(
            "deterministic repair set mismatch: "
            f"expected={sorted(expected_repairs)}, "
            f"found={sorted(repaired)}"
        )
    reference: dict[str, object] = {}
    if compare_reference and REFERENCE_ID in expected:
        generated_joints = load_npy(
            checked_path(joints_root, REFERENCE_ID),
            validate_joints_array,
        )
        generated_vectors = load_npy(
            checked_path(vectors_root, REFERENCE_ID),
            validate_vector_array,
        )
        reference["new_joints"] = compare_reference_array(
            generated_joints,
            source_root / f"HumanML3D/new_joints/{REFERENCE_ID}",
            validate_joints_array,
            rtol=2.0e-4,
            atol=2.0e-5,
            expected_sha256=REFERENCE_HASHES["new_joints"],
        )
        reference["new_joint_vecs"] = compare_reference_array(
            generated_vectors,
            source_root / f"HumanML3D/new_joint_vecs/{REFERENCE_ID}",
            validate_vector_array,
            rtol=2.0e-4,
            atol=2.0e-5,
            expected_sha256=REFERENCE_HASHES["new_joint_vecs"],
        )
    if enforce_official_counts:
        if len(expected) != OFFICIAL_REPRESENT_FILES:
            raise PipelineError("representation file count gate failed")
        if total_frames != OFFICIAL_REPRESENT_FRAMES:
            raise PipelineError(
                f"representation frames: expected "
                f"{OFFICIAL_REPRESENT_FRAMES}, found {total_frames}"
            )
        all_ids = {
            f"{line.strip()}.npy"
            for line in (
                source_root / "HumanML3D/all.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if all_ids != expected:
            raise PipelineError(
                "official all.txt does not match represented file set"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "represent",
        "verified_at": utc_now(),
        "fingerprint": fingerprint,
        "files": len(expected),
        "frames": total_frames,
        "finite": True,
        "feature_dim": FEATURE_DIM,
        "repaired_nonfinite_ids": sorted(repaired),
        "repair_details": {
            name: dict(value)
            for name, value in sorted(repair_details.items())
        },
        "official_difference": {
            "ids": sorted(repaired),
            "official_behavior": (
                "official notebook emitted non-finite arrays and excluded "
                "these IDs from Mean/Std"
            ),
            "replacement": (
                "epsilon-safe vector normalization, previous-valid "
                "orientation fallback, deterministic antiparallel axis, "
                "and clipped arcsin"
            ),
            "epsilon": REPAIR_EPSILON,
        },
        "expected_short_ids": sorted(
            set(EXPECTED_SHORT_IDS) & contract.segment_names
        ),
        "reference_parity": reference,
        "record_set_sha256": canonical_sha256(
            sorted(ledger, key=lambda item: str(item["output"]))
        ),
    }


def represent_dataset(
    contract: IndexContract,
    source_root: Path,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    space_guard: SpaceGuard,
    *,
    verify_resume_hash: bool,
    enforce_official_counts: bool,
    compare_reference: bool,
) -> dict[str, object]:
    backend = load_official_backend(source_root)
    joints_root = checked_path(staging_root, "joints")
    target_path = checked_path(joints_root, TARGET_SKELETON_ID)
    target_sha256 = sha256_file(target_path)
    target_offsets = target_offsets_from_file(target_path, backend)
    new_joints_root = checked_path(
        staging_root, "HumanML3D/new_joints"
    )
    vectors_root = checked_path(
        staging_root, "HumanML3D/new_joint_vecs"
    )
    new_joints_root.mkdir(parents=True, exist_ok=True)
    vectors_root.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temps(
        new_joints_root, set(contract.represent_names)
    )
    cleanup_atomic_temps(vectors_root, set(contract.represent_names))
    repair_details: dict[str, Mapping[str, int]] = {}
    completed = 0
    for name in sorted(contract.segment_names):
        source_path = checked_path(joints_root, name)
        source = None
        if name in EXPECTED_SHORT_IDS:
            source = load_npy(source_path, validate_segment_array)
            if source.shape[0] != 1:
                raise PipelineError(
                    f"{name}: expected one frame, found {source.shape[0]}"
                )
            for root in (new_joints_root, vectors_root):
                if (root / name).exists():
                    raise PipelineError(
                        f"{name}: stale representation exists for skip"
                    )
            continue
        source_sha256 = sha256_file(source_path)
        item_fingerprint = canonical_sha256(
            {
                "stage": fingerprint,
                "source": name,
                "source_sha256": source_sha256,
                "target_sha256": target_sha256,
                "repair": name in REPAIRED_NONFINITE_IDS,
                "epsilon": REPAIR_EPSILON,
            }
        )
        joints_rel = f"HumanML3D/new_joints/{name}"
        vectors_rel = f"HumanML3D/new_joint_vecs/{name}"
        joints_complete = state.complete(
            "represent",
            joints_rel,
            f"{fingerprint}:{item_fingerprint}:joints",
            joints_rel,
            validate_joints_array,
            verify_hash=verify_resume_hash,
        )
        vectors_complete = state.complete(
            "represent",
            vectors_rel,
            f"{fingerprint}:{item_fingerprint}:vectors",
            vectors_rel,
            validate_vector_array,
            verify_hash=verify_resume_hash,
        )
        repair = name in REPAIRED_NONFINITE_IDS
        if joints_complete and vectors_complete and not repair:
            completed += 1
            continue
        if source is None:
            source = load_npy(source_path, validate_segment_array)
        source = source[:, :JOINTS_NUM]
        recovered, vectors, diagnostics = process_motion(
            source,
            target_offsets,
            backend,
            repair=repair,
            epsilon=REPAIR_EPSILON,
        )
        if repair:
            repair_details[name] = diagnostics
        joints_info = atomic_npy(
            checked_path(staging_root, joints_rel),
            recovered,
            validate_joints_array,
            space_guard=space_guard,
            expected_frames=source.shape[0] - 1,
        )
        vectors_info = atomic_npy(
            checked_path(staging_root, vectors_rel),
            vectors,
            validate_vector_array,
            space_guard=space_guard,
            expected_frames=source.shape[0] - 1,
        )
        state.record(
            "represent",
            joints_rel,
            f"{fingerprint}:{item_fingerprint}:joints",
            joints_rel,
            joints_info,
        )
        state.record(
            "represent",
            vectors_rel,
            f"{fingerprint}:{item_fingerprint}:vectors",
            vectors_rel,
            vectors_info,
        )
        completed += 1
        if completed % 100 == 0:
            print(
                f"REPRESENT {completed}/{len(contract.represent_names)}",
                flush=True,
            )
    result = verify_representation_tree(
        contract,
        source_root,
        staging_root,
        state,
        fingerprint,
        repair_details,
        verify_hash=True,
        enforce_official_counts=enforce_official_counts,
        compare_reference=compare_reference,
    )
    atomic_json(
        manifest_path(
            staging_root, "postprocess-represent-success.json"
        ),
        result,
    )
    return result


class OnlineMoments:
    """Numerically stable, bounded-memory population moments."""

    def __init__(self, width: int):
        np = import_numpy()
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values) -> None:
        np = import_numpy()
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.shape[0]:
            raise PipelineError(f"moment batch has shape {batch.shape}")
        if not np.isfinite(batch).all():
            raise PipelineError("moment batch contains NaN or Inf")
        batch_count = int(batch.shape[0])
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0, dtype=np.float64)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / total)
        )
        self.count = total

    def finalize(self):
        np = import_numpy()
        if self.count < 1:
            raise PipelineError("cannot finalize empty statistics")
        variance = np.maximum(self.m2 / self.count, 0.0)
        mean = self.mean.astype(np.float32)
        std = np.sqrt(variance).astype(np.float32)
        for start, end in FEATURE_GROUPS:
            std[start:end] = std[start:end].mean(dtype=np.float32)
        validate_stat_array(mean, label="Mean")
        validate_stat_array(std, label="Std")
        if np.any(std <= 0):
            raise PipelineError("Std contains a non-positive group")
        return mean, std


def available_memory_bytes() -> int:
    try:
        lines = Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines()
        for line in lines:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) * 1024
    except (OSError, UnicodeError, ValueError):
        pass
    try:
        return (
            int(os.sysconf("SC_AVPHYS_PAGES"))
            * int(os.sysconf("SC_PAGE_SIZE"))
        )
    except (OSError, ValueError) as exc:
        raise PipelineError(
            "cannot determine available memory for legacy statistics"
        ) from exc


def legacy_memory_gate(
    input_bytes: int, *, available_bytes: int | None = None
) -> dict[str, int]:
    if input_bytes <= 0:
        raise PipelineError("legacy statistics input is empty")
    available = (
        available_memory_bytes()
        if available_bytes is None
        else int(available_bytes)
    )
    required = (
        input_bytes * LEGACY_RAM_MULTIPLIER
        + LEGACY_RAM_SAFETY_BYTES
    )
    if available < required:
        raise PipelineError(
            "insufficient RAM for official float32 concatenate: "
            f"required={required}, available={available}"
        )
    return {
        "input_bytes": input_bytes,
        "required_available_bytes": required,
        "available_bytes_at_gate": available,
    }


def array_difference(left, right) -> dict[str, float]:
    np = import_numpy()
    difference = np.abs(
        left.astype(np.float64) - right.astype(np.float64)
    )
    return {
        "max_abs": (
            float(difference.max()) if difference.size else 0.0
        ),
        "mean_abs": (
            float(difference.mean()) if difference.size else 0.0
        ),
    }


def legacy_float32_statistics(
    vectors_root: Path,
    expected_names: set[str],
    excluded_names: set[str],
    *,
    available_bytes: int | None = None,
):
    """Reproduce the official notebook's order and float32 reductions."""
    np = import_numpy()
    observed, unsafe = list_relative_files(vectors_root)
    if unsafe or observed != expected_names:
        raise PipelineError(
            "legacy statistics file set mismatch: "
            f"unsafe={unsafe[:5]}, "
            f"missing={sorted(expected_names - observed)[:5]}, "
            f"extra={sorted(observed - expected_names)[:5]}"
        )
    directory_order = os.listdir(vectors_root)
    if (
        len(directory_order) != len(expected_names)
        or set(directory_order) != expected_names
    ):
        raise PipelineError(
            "legacy statistics directory order is not the exact file set"
        )
    if not excluded_names.issubset(expected_names):
        raise PipelineError("legacy statistics exclusions are invalid")
    effective_order = [
        name for name in directory_order if name not in excluded_names
    ]
    input_bytes = sum(
        checked_path(vectors_root, name).stat().st_size
        for name in effective_order
    )
    memory = legacy_memory_gate(
        input_bytes, available_bytes=available_bytes
    )
    try:
        parts = [
            load_npy(
                checked_path(vectors_root, name),
                validate_vector_array,
            )
            for name in effective_order
        ]
        data = np.concatenate(parts, axis=0)
    except MemoryError as exc:
        raise PipelineError(
            "official float32 concatenate exhausted RAM despite gate"
        ) from exc
    del parts
    frames = int(data.shape[0])
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    del data
    for start, end in FEATURE_GROUPS:
        std[start:end] = std[start:end].mean(dtype=np.float32)
    validate_stat_array(mean, label="official-compatible Mean")
    validate_stat_array(std, label="official-compatible Std")
    if np.any(std <= 0):
        raise PipelineError(
            "official-compatible Std contains a non-positive group"
        )
    return mean, std, {
        "reduction": "np.concatenate float32 population mean/std",
        "order": "os.listdir",
        "order_sha256": canonical_sha256(effective_order),
        "files": len(effective_order),
        "frames": frames,
        "excluded_repaired_ids": sorted(excluded_names),
        "accumulator_dtype": "float32",
        "output_dtype": "float32",
        "ddof": 0,
        "memory_gate": memory,
    }
def verify_ordered_id_file(
    path: Path, expected: Sequence[str], *, label: str
) -> str:
    if not path.is_file() or path.is_symlink():
        raise PipelineError(f"{label} is missing or unsafe: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"cannot read {label}: {exc}") from exc
    expected_text = "".join(f"{value}\n" for value in expected)
    if text != expected_text:
        raise PipelineError(
            f"{label} is not the exact ordered ID contract"
        )
    return sha256_file(path)


def verify_statistics_files(
    contract: IndexContract,
    source_root: Path,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    *,
    verify_hash: bool,
    compare_reference: bool,
) -> dict[str, object]:
    np = import_numpy()
    names = (
        "Mean.npy",
        "Std.npy",
        "Mean_stable.npy",
        "Std_stable.npy",
        "Mean_all_finite.npy",
        "Std_all_finite.npy",
    )
    records = state.records("stats")
    expected_keys = {f"HumanML3D/{name}" for name in names}
    if set(records) != expected_keys:
        raise PipelineError(
            "stats ledger mismatch: "
            f"missing={sorted(expected_keys - set(records))}, "
            f"extra={sorted(set(records) - expected_keys)}"
        )
    arrays: dict[str, object] = {}
    ledger: list[dict[str, str]] = []
    for name in names:
        output_rel = f"HumanML3D/{name}"
        path = checked_path(staging_root, output_rel)
        arrays[name] = load_npy(path, validate_stat_array)
        record = records[output_rel]
        if (
            record["output_rel"] != output_rel
            or record["fingerprint"] != f"{fingerprint}:{name}"
            or path.stat().st_size != record["output_size"]
        ):
            raise PipelineError(f"{output_rel}: stats ledger mismatch")
        digest = str(record["output_sha256"])
        if verify_hash and sha256_file(path) != digest:
            raise PipelineError(f"{output_rel}: stats digest mismatch")
        ledger.append({"output": output_rel, "sha256": digest})
    for name in (
        "Std.npy",
        "Std_stable.npy",
        "Std_all_finite.npy",
    ):
        array = arrays[name]
        for start, end in FEATURE_GROUPS:
            if not np.all(array[start:end] == array[start]):
                raise PipelineError(
                    f"{name}: group {start}:{end} is not shared"
                )
    reference: dict[str, object] = {}
    if compare_reference:
        reference["Mean"] = compare_reference_array(
            arrays["Mean.npy"],
            source_root / "HumanML3D/Mean.npy",
            validate_stat_array,
            rtol=0.0,
            atol=LEGACY_MEAN_REFERENCE_ATOL,
            expected_sha256=REFERENCE_HASHES["Mean"],
        )
        reference["Std"] = compare_reference_array(
            arrays["Std.npy"],
            source_root / "HumanML3D/Std.npy",
            validate_stat_array,
            rtol=0.0,
            atol=LEGACY_STD_REFERENCE_ATOL,
            expected_sha256=REFERENCE_HASHES["Std"],
        )
    finite_ids = checked_path(staging_root, "HumanML3D/finite_ids.txt")
    excluded_ids = checked_path(
        staging_root, "HumanML3D/compat_stats_excluded_ids.txt"
    )
    finite_expected = [
        name.removesuffix(".npy")
        for name in sorted(contract.represent_names)
    ]
    excluded_expected = [
        name.removesuffix(".npy")
        for name in sorted(
            set(REPAIRED_NONFINITE_IDS)
            & set(contract.represent_names)
        )
    ]
    finite_ids_sha256 = verify_ordered_id_file(
        finite_ids, finite_expected, label="finite IDs"
    )
    excluded_ids_sha256 = verify_ordered_id_file(
        excluded_ids,
        excluded_expected,
        label="compatibility-excluded IDs",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "stats",
        "verified_at": utc_now(),
        "fingerprint": fingerprint,
        "finite": True,
        "files": len(names),
        "record_set_sha256": canonical_sha256(ledger),
        "reference_parity": reference,
        "finite_ids_sha256": finite_ids_sha256,
        "compat_excluded_ids_sha256": excluded_ids_sha256,
    }


def stats_dataset(
    contract: IndexContract,
    source_root: Path,
    staging_root: Path,
    state: StateDB,
    fingerprint: str,
    space_guard: SpaceGuard,
    *,
    verify_resume_hash: bool,
    enforce_official_counts: bool,
    compare_reference: bool,
) -> dict[str, object]:
    output_root = checked_path(staging_root, "HumanML3D")
    output_root.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temps(
        output_root,
        {
            "Mean.npy",
            "Std.npy",
            "Mean_stable.npy",
            "Std_stable.npy",
            "Mean_all_finite.npy",
            "Std_all_finite.npy",
            "finite_ids.txt",
            "compat_stats_excluded_ids.txt",
        },
    )
    vectors_root = checked_path(
        staging_root, "HumanML3D/new_joint_vecs"
    )
    stable_compatibility = OnlineMoments(FEATURE_DIM)
    all_finite = OnlineMoments(FEATURE_DIM)
    compatibility_files = 0
    all_files = 0
    for name in sorted(contract.represent_names):
        values = load_npy(
            checked_path(vectors_root, name), validate_vector_array
        )
        all_finite.update(values)
        all_files += 1
        if name not in REPAIRED_NONFINITE_IDS:
            stable_compatibility.update(values)
            compatibility_files += 1
    mean_stable, std_stable = stable_compatibility.finalize()
    mean_all, std_all = all_finite.finalize()
    expected_names = set(contract.represent_names)
    excluded_names = set(REPAIRED_NONFINITE_IDS) & expected_names
    mean_legacy, std_legacy, legacy_contract = (
        legacy_float32_statistics(
            vectors_root,
            expected_names,
            excluded_names,
        )
    )
    if enforce_official_counts:
        gates = (
            (
                all_files,
                OFFICIAL_REPRESENT_FILES,
                "all-finite stats files",
            ),
            (
                all_finite.count,
                OFFICIAL_REPRESENT_FRAMES,
                "all-finite stats frames",
            ),
            (
                compatibility_files,
                OFFICIAL_COMPAT_STATS_FILES,
                "compat stats files",
            ),
            (
                stable_compatibility.count,
                OFFICIAL_COMPAT_STATS_FRAMES,
                "compat stats frames",
            ),
            (
                legacy_contract["files"],
                OFFICIAL_COMPAT_STATS_FILES,
                "legacy compat stats files",
            ),
            (
                legacy_contract["frames"],
                OFFICIAL_COMPAT_STATS_FRAMES,
                "legacy compat stats frames",
            ),
        )
        for observed, expected, label in gates:
            if observed != expected:
                raise PipelineError(
                    f"{label}: expected {expected}, found {observed}"
                )
    outputs = {
        "Mean.npy": mean_legacy,
        "Std.npy": std_legacy,
        "Mean_stable.npy": mean_stable,
        "Std_stable.npy": std_stable,
        "Mean_all_finite.npy": mean_all,
        "Std_all_finite.npy": std_all,
    }
    for name, array in outputs.items():
        output_rel = f"HumanML3D/{name}"
        info = atomic_npy(
            checked_path(staging_root, output_rel),
            array,
            validate_stat_array,
            space_guard=space_guard,
        )
        state.record(
            "stats",
            output_rel,
            f"{fingerprint}:{name}",
            output_rel,
            info,
        )
    finite_text = "".join(
        f"{name.removesuffix('.npy')}\n"
        for name in sorted(contract.represent_names)
    )
    excluded = sorted(
        set(REPAIRED_NONFINITE_IDS) & set(contract.represent_names)
    )
    excluded_text = "".join(
        f"{name.removesuffix('.npy')}\n" for name in excluded
    )
    atomic_text(
        checked_path(staging_root, "HumanML3D/finite_ids.txt"),
        finite_text,
    )
    atomic_text(
        checked_path(
            staging_root, "HumanML3D/compat_stats_excluded_ids.txt"
        ),
        excluded_text,
    )
    verified = verify_statistics_files(
        contract,
        source_root,
        staging_root,
        state,
        fingerprint,
        verify_hash=True,
        compare_reference=compare_reference,
    )
    result = {
        **verified,
        "primary_compatibility": {
            "mean": "HumanML3D/Mean.npy",
            "std": "HumanML3D/Std.npy",
            **legacy_contract,
            "purpose": "official HumanML3D normalization compatibility",
        },
        "stable_compatibility": {
            "mean": "HumanML3D/Mean_stable.npy",
            "std": "HumanML3D/Std_stable.npy",
            "files": compatibility_files,
            "frames": stable_compatibility.count,
            "excluded_repaired_ids": excluded,
            "reduction": "batch-merged Welford population moments",
            "accumulator_dtype": "float64",
            "output_dtype": "float32",
            "ddof": 0,
            "purpose": "numerically stable normalization excluding repaired clips",
        },
        "all_finite": {
            "mean": "HumanML3D/Mean_all_finite.npy",
            "std": "HumanML3D/Std_all_finite.npy",
            "files": all_files,
            "frames": all_finite.count,
            "purpose": "normalization including deterministic repairs",
        },
    }
    atomic_json(
        manifest_path(staging_root, "postprocess-stats-success.json"),
        result,
    )
    return result


@dataclasses.dataclass(frozen=True)
class ProductionContext:
    contract: IndexContract
    index_sha256: str
    algorithm: Mapping[str, object]
    pose_success: Mapping[str, object]
    pose_success_sha256: str
    pose_hashes: Mapping[str, str]
    segment_fingerprint: str


def pose_binding(value: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "pose_fingerprint": value["pose_fingerprint"],
            "record_set_sha256": value["record_set_sha256"],
            "index_sha256": value["index_sha256"],
            "sources": value["sources"],
            "total_frames": value.get("total_frames"),
            "finite": value["finite"],
            "joint_floor": value["joint_floor"],
        }
    )


def marker_binding(value: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "stage": value["stage"],
            "fingerprint": value["fingerprint"],
            "record_set_sha256": value["record_set_sha256"],
            "files": value["files"],
            "frames": value.get("frames"),
            "finite": value["finite"],
            "finite_ids_sha256": value.get("finite_ids_sha256"),
            "compat_excluded_ids_sha256": value.get(
                "compat_excluded_ids_sha256"
            ),
        }
    )


def prepare_production(args: argparse.Namespace) -> ProductionContext:
    validate_layout(args.source_root, args.staging_root, args.data_root)
    force_cpu_environment()
    index_path = args.source_root / "index.csv"
    index_sha256 = sha256_file(index_path)
    contract = load_index_contract(index_path)
    algorithm = algorithm_contract(args.source_root)
    pose_success, pose_success_sha256 = load_pose_success(
        args.staging_root, index_sha256, enforce_official_counts=True
    )
    pose_hashes = load_pose_ledger(
        args.staging_root, contract, pose_success
    )
    segment_fingerprint = stage_fingerprint(
        "segment",
        algorithm,
        index_sha256,
        pose_binding(pose_success),
    )
    return ProductionContext(
        contract=contract,
        index_sha256=index_sha256,
        algorithm=algorithm,
        pose_success=pose_success,
        pose_success_sha256=pose_success_sha256,
        pose_hashes=pose_hashes,
        segment_fingerprint=segment_fingerprint,
    )


def load_segment_marker(
    context: ProductionContext, staging_root: Path
) -> Mapping[str, object]:
    return load_stage_success(
        manifest_path(
            staging_root, "postprocess-segment-success.json"
        ),
        "segment",
        context.segment_fingerprint,
    )


def representation_fingerprint(
    context: ProductionContext, segment: Mapping[str, object]
) -> str:
    return stage_fingerprint(
        "represent",
        context.algorithm,
        context.index_sha256,
        marker_binding(segment),
    )


def load_representation_marker(
    context: ProductionContext,
    staging_root: Path,
    segment: Mapping[str, object],
) -> tuple[Mapping[str, object], str]:
    fingerprint = representation_fingerprint(context, segment)
    marker = load_stage_success(
        manifest_path(
            staging_root, "postprocess-represent-success.json"
        ),
        "represent",
        fingerprint,
    )
    repaired = set(marker.get("repaired_nonfinite_ids", ()))
    if repaired != set(REPAIRED_NONFINITE_IDS):
        raise PipelineError("representation repair marker mismatch")
    return marker, fingerprint


def statistics_fingerprint(
    context: ProductionContext, representation: Mapping[str, object]
) -> str:
    statistics = context.algorithm.get("statistics")
    if not isinstance(statistics, Mapping):
        raise PipelineError("statistics algorithm contract is missing")
    statistics_sha256 = statistics.get("fingerprint")
    validate_sha256(
        statistics_sha256, label="statistics algorithm fingerprint"
    )
    algorithm = {
        "fingerprint": canonical_sha256(
            {
                "upstream": context.algorithm["fingerprint"],
                "statistics": statistics_sha256,
            }
        )
    }
    return stage_fingerprint(
        "stats",
        algorithm,
        context.index_sha256,
        marker_binding(representation),
    )


def load_statistics_marker(
    context: ProductionContext,
    staging_root: Path,
    representation: Mapping[str, object],
) -> tuple[Mapping[str, object], str]:
    fingerprint = statistics_fingerprint(context, representation)
    marker = load_stage_success(
        manifest_path(staging_root, "postprocess-stats-success.json"),
        "stats",
        fingerprint,
    )
    primary = marker.get("primary_compatibility")
    stable = marker.get("stable_compatibility")
    all_finite = marker.get("all_finite")
    if (
        not isinstance(primary, Mapping)
        or not isinstance(stable, Mapping)
        or not isinstance(all_finite, Mapping)
    ):
        raise PipelineError("statistics contracts are missing")
    if (
        primary.get("files") != OFFICIAL_COMPAT_STATS_FILES
        or primary.get("frames") != OFFICIAL_COMPAT_STATS_FRAMES
        or set(primary.get("excluded_repaired_ids", ()))
        != set(REPAIRED_NONFINITE_IDS)
    ):
        raise PipelineError("compatibility statistics count gate failed")
    if (
        stable.get("files") != OFFICIAL_COMPAT_STATS_FILES
        or stable.get("frames") != OFFICIAL_COMPAT_STATS_FRAMES
        or set(stable.get("excluded_repaired_ids", ()))
        != set(REPAIRED_NONFINITE_IDS)
    ):
        raise PipelineError("stable statistics count gate failed")
    if (
        all_finite.get("files") != OFFICIAL_REPRESENT_FILES
        or all_finite.get("frames") != OFFICIAL_REPRESENT_FRAMES
    ):
        raise PipelineError("all-finite statistics count gate failed")
    return marker, fingerprint


def stage_locks(staging_root: Path):
    return (
        FileLock(
            checked_path(staging_root, ".pipeline.lock"),
            shared=True,
            blocking=False,
        ),
        FileLock(
            checked_path(staging_root, ".postprocess.lock"),
            shared=False,
            blocking=False,
        ),
    )


@contextlib.contextmanager
def locked_production(args: argparse.Namespace):
    """Hold the pose shared lock and postprocess exclusive lock."""
    validate_layout(args.source_root, args.staging_root, args.data_root)
    pose_lock, own_lock = stage_locks(args.staging_root)
    with pose_lock, own_lock:
        yield


def make_space_guard(
    staging_root: Path, reserve_gib: float
) -> SpaceGuard:
    if (
        not math.isfinite(reserve_gib)
        or reserve_gib < 0
        or reserve_gib >= 10_000
    ):
        raise PipelineError("invalid reserve-gib")
    guard = SpaceGuard(staging_root, int(reserve_gib * 1024**3))
    guard.require(label="postprocess startup")
    return guard


def run_segment(args: argparse.Namespace) -> None:
    with locked_production(args):
        context = prepare_production(args)
        invalidate_stage_markers(args.staging_root, "segment")
        configure_runtime(args.staging_root)
        guard = make_space_guard(args.staging_root, args.reserve_gib)
        state_path = checked_path(
            args.staging_root, "postprocess_state.sqlite3"
        )
        with StateDB(state_path, args.staging_root) as state:
            result = segment_dataset(
                context.contract,
                checked_path(args.staging_root, "pose_data"),
                args.staging_root,
                state,
                context.segment_fingerprint,
                context.pose_hashes,
                guard,
                verify_resume_hash=args.verify_resume_hash,
                enforce_official_counts=True,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_represent(args: argparse.Namespace) -> None:
    with locked_production(args):
        context = prepare_production(args)
        invalidate_stage_markers(args.staging_root, "represent")
        segment = load_segment_marker(context, args.staging_root)
        fingerprint = representation_fingerprint(context, segment)
        configure_runtime(args.staging_root)
        guard = make_space_guard(args.staging_root, args.reserve_gib)
        state_path = checked_path(
            args.staging_root, "postprocess_state.sqlite3"
        )
        with StateDB(state_path, args.staging_root) as state:
            result = represent_dataset(
                context.contract,
                args.source_root,
                args.staging_root,
                state,
                fingerprint,
                guard,
                verify_resume_hash=args.verify_resume_hash,
                enforce_official_counts=True,
                compare_reference=args.compare_reference,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_stats(args: argparse.Namespace) -> None:
    with locked_production(args):
        context = prepare_production(args)
        invalidate_stage_markers(args.staging_root, "stats")
        segment = load_segment_marker(context, args.staging_root)
        representation, _ = load_representation_marker(
            context, args.staging_root, segment
        )
        fingerprint = statistics_fingerprint(context, representation)
        configure_runtime(args.staging_root)
        guard = make_space_guard(args.staging_root, args.reserve_gib)
        state_path = checked_path(
            args.staging_root, "postprocess_state.sqlite3"
        )
        with StateDB(state_path, args.staging_root) as state:
            result = stats_dataset(
                context.contract,
                args.source_root,
                args.staging_root,
                state,
                fingerprint,
                guard,
                verify_resume_hash=args.verify_resume_hash,
                enforce_official_counts=True,
                compare_reference=args.compare_reference,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def verify_all(
    context: ProductionContext,
    source_root: Path,
    staging_root: Path,
    state: StateDB,
    *,
    verify_hash: bool,
    compare_reference: bool,
) -> dict[str, object]:
    segment_marker = load_segment_marker(context, staging_root)
    representation_marker, representation_fp = (
        load_representation_marker(
            context, staging_root, segment_marker
        )
    )
    statistics_marker, statistics_fp = load_statistics_marker(
        context, staging_root, representation_marker
    )
    repair_details = representation_marker.get("repair_details")
    if not isinstance(repair_details, Mapping) or any(
        not isinstance(value, Mapping)
        for value in repair_details.values()
    ):
        raise PipelineError("representation repair details are invalid")
    segment = verify_segment_tree(
        context.contract,
        staging_root,
        state,
        context.segment_fingerprint,
        verify_hash=verify_hash,
        enforce_official_counts=True,
    )
    representation = verify_representation_tree(
        context.contract,
        source_root,
        staging_root,
        state,
        representation_fp,
        repair_details,
        verify_hash=verify_hash,
        enforce_official_counts=True,
        compare_reference=compare_reference,
    )
    statistics = verify_statistics_files(
        context.contract,
        source_root,
        staging_root,
        state,
        statistics_fp,
        verify_hash=verify_hash,
        compare_reference=compare_reference,
    )
    for observed, marker, label in (
        (segment, segment_marker, "segment"),
        (representation, representation_marker, "representation"),
        (statistics, statistics_marker, "statistics"),
    ):
        if observed["record_set_sha256"] != marker["record_set_sha256"]:
            raise PipelineError(f"{label} record set changed")
        if label == "statistics":
            for field in (
                "finite_ids_sha256",
                "compat_excluded_ids_sha256",
            ):
                if observed[field] != marker.get(field):
                    raise PipelineError(
                        f"statistics {field} changed"
                    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "verify",
        "verified_at": utc_now(),
        "finite": True,
        "segment": {
            "files": segment["files"],
            "frames": segment["frames"],
            "record_set_sha256": segment["record_set_sha256"],
        },
        "representation": {
            "files": representation["files"],
            "frames": representation["frames"],
            "record_set_sha256": representation[
                "record_set_sha256"
            ],
            "repaired_nonfinite_ids": representation[
                "repaired_nonfinite_ids"
            ],
            "reference_parity": representation["reference_parity"],
        },
        "statistics": {
            "record_set_sha256": statistics["record_set_sha256"],
            "reference_parity": statistics["reference_parity"],
        },
        "source_bindings": {
            "index_sha256": context.index_sha256,
            "pose_record_set_sha256": context.pose_success[
                "record_set_sha256"
            ],
            "algorithm_fingerprint": context.algorithm["fingerprint"],
        },
    }
    payload["fingerprint"] = canonical_sha256(
        {
            "segment": marker_binding(segment_marker),
            "representation": marker_binding(representation_marker),
            "statistics": marker_binding(statistics_marker),
            "source_bindings": payload["source_bindings"],
        }
    )
    return payload


def verify_splits_and_texts(
    contract: IndexContract, source_root: Path, staging_root: Path
) -> dict[str, object]:
    metadata_root = source_root / "HumanML3D"
    expected = {
        name.removesuffix(".npy") for name in contract.represent_names
    }
    expected_counts = {
        "train.txt": 23_384,
        "val.txt": 1_460,
        "test.txt": 4_384,
        "train_val.txt": 24_844,
        "all.txt": 29_228,
    }
    splits: dict[str, set[str]] = {}
    for filename, expected_count in expected_counts.items():
        path = metadata_root / filename
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(values) != len(set(values)):
            raise PipelineError(f"{filename}: duplicate IDs")
        if len(values) != expected_count:
            raise PipelineError(
                f"{filename}: expected {expected_count}, "
                f"found {len(values)}"
            )
        split = set(values)
        if not split.issubset(expected):
            raise PipelineError(
                f"{filename}: IDs missing finite representations"
            )
        splits[filename] = split
    if splits["all.txt"] != expected:
        raise PipelineError("all.txt is not the finite representation set")
    if (
        splits["train.txt"] & splits["val.txt"]
        or splits["train.txt"] & splits["test.txt"]
        or splits["val.txt"] & splits["test.txt"]
    ):
        raise PipelineError("train/val/test splits overlap")
    if (
        splits["train.txt"]
        | splits["val.txt"]
        | splits["test.txt"]
    ) != expected:
        raise PipelineError("train/val/test union is not all.txt")
    if splits["train_val.txt"] != (
        splits["train.txt"] | splits["val.txt"]
    ):
        raise PipelineError("train_val.txt is not train union val")
    text_root = metadata_root / "texts"
    missing_text = [
        name
        for name in sorted(expected)
        if not (text_root / f"{name}.txt").is_file()
        or (text_root / f"{name}.txt").is_symlink()
    ]
    if missing_text:
        raise PipelineError(
            f"finite IDs missing text files: {missing_text[:5]}"
        )
    finite_order = sorted(expected)
    finite_ids_sha256 = verify_ordered_id_file(
        checked_path(staging_root, "HumanML3D/finite_ids.txt"),
        finite_order,
        label="finite IDs",
    )
    excluded_order = sorted(
        name.removesuffix(".npy") for name in REPAIRED_NONFINITE_IDS
    )
    excluded_ids_sha256 = verify_ordered_id_file(
        checked_path(
            staging_root, "HumanML3D/compat_stats_excluded_ids.txt"
        ),
        excluded_order,
        label="compatibility-excluded IDs",
    )
    return {
        "metadata_root": str(metadata_root),
        "text_root": str(text_root),
        "split_counts": expected_counts,
        "finite_texts": len(expected),
        "finite_ids_sha256": finite_ids_sha256,
        "compat_excluded_ids_sha256": excluded_ids_sha256,
    }


def run_verify(args: argparse.Namespace) -> None:
    with locked_production(args):
        context = prepare_production(args)
        invalidate_stage_markers(args.staging_root, "verify")
        configure_runtime(args.staging_root)
        state_path = checked_path(
            args.staging_root, "postprocess_state.sqlite3"
        )
        with StateDB(
            state_path,
            args.staging_root,
        ) as state:
            result = verify_all(
                context,
                args.source_root,
                args.staging_root,
                state,
                verify_hash=True,
                compare_reference=args.compare_reference,
            )
        atomic_json(
            manifest_path(
                args.staging_root,
                "postprocess-verify-success.json",
            ),
            result,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_ready(args: argparse.Namespace) -> None:
    if not args.compare_reference:
        raise PipelineError(
            "ready requires official reference comparisons"
        )
    with locked_production(args):
        context = prepare_production(args)
        invalidate_stage_markers(args.staging_root, "ready")
        configure_runtime(args.staging_root)
        state_path = checked_path(
            args.staging_root, "postprocess_state.sqlite3"
        )
        with StateDB(
            state_path,
            args.staging_root,
        ) as state:
            verification = verify_all(
                context,
                args.source_root,
                args.staging_root,
                state,
                verify_hash=True,
                compare_reference=args.compare_reference,
            )
        metadata = verify_splits_and_texts(
            context.contract, args.source_root, args.staging_root
        )
        verify_path = manifest_path(
            args.staging_root,
            "postprocess-verify-success.json",
        )
        atomic_json(verify_path, verification)
        segment_path = manifest_path(
            args.staging_root,
            "postprocess-segment-success.json",
        )
        representation_path = manifest_path(
            args.staging_root,
            "postprocess-represent-success.json",
        )
        stats_path = manifest_path(
            args.staging_root,
            "postprocess-stats-success.json",
        )
        ready = {
            "schema_version": SCHEMA_VERSION,
            "stage": "ready",
            "ready_at": utc_now(),
            "ready": True,
            "finite": True,
            "staging_root": str(args.staging_root.resolve()),
            "motion_root": str(
                checked_path(args.staging_root, "HumanML3D").resolve()
            ),
            "metadata": metadata,
            "counts": {
                "new_joints": OFFICIAL_REPRESENT_FILES,
                "new_joint_vecs": OFFICIAL_REPRESENT_FILES,
                "frames": OFFICIAL_REPRESENT_FRAMES,
            },
            "repaired_nonfinite_ids": sorted(
                REPAIRED_NONFINITE_IDS
            ),
            "repair_difference_from_official": (
                "official notebook produced NaN for these two IDs; "
                "this build uses deterministic epsilon/orientation fallback"
            ),
            "normalization": {
                "compatible": {
                    "mean": "HumanML3D/Mean.npy",
                    "std": "HumanML3D/Std.npy",
                    "excludes": sorted(REPAIRED_NONFINITE_IDS),
                    "use": "official-model compatibility",
                },
                "stable": {
                    "mean": "HumanML3D/Mean_stable.npy",
                    "std": "HumanML3D/Std_stable.npy",
                    "excludes": sorted(REPAIRED_NONFINITE_IDS),
                    "use": "numerically stable normalization",
                },
                "all_finite": {
                    "mean": "HumanML3D/Mean_all_finite.npy",
                    "std": "HumanML3D/Std_all_finite.npy",
                    "includes": sorted(REPAIRED_NONFINITE_IDS),
                    "use": "training that includes repaired clips",
                },
            },
            "bindings": {
                "pose_success_sha256": context.pose_success_sha256,
                "segment_success_sha256": sha256_file(segment_path),
                "represent_success_sha256": sha256_file(
                    representation_path
                ),
                "stats_success_sha256": sha256_file(stats_path),
                "verify_success_sha256": sha256_file(verify_path),
                "index_sha256": context.index_sha256,
                "algorithm_fingerprint": context.algorithm[
                    "fingerprint"
                ],
            },
        }
        ready["fingerprint"] = canonical_sha256(
            {
                "bindings": ready["bindings"],
                "counts": ready["counts"],
                "normalization": ready["normalization"],
                "repaired_nonfinite_ids": ready[
                    "repaired_nonfinite_ids"
                ],
            }
        )
        ready_path = manifest_path(
            args.staging_root, "postprocess-ready.json"
        )
        atomic_json(ready_path, ready)
    print(f"POSTPROCESS_READY {ready_path}")
    print(json.dumps(ready, ensure_ascii=False, sort_keys=True))


def run_status(args: argparse.Namespace) -> None:
    root = args.staging_root
    manifests = (
        "pose-success.json",
        "postprocess-segment-success.json",
        "postprocess-represent-success.json",
        "postprocess-stats-success.json",
        "postprocess-verify-success.json",
        "postprocess-ready.json",
    )
    counts: dict[str, int] = {}
    state_path = root / "postprocess_state.sqlite3"
    if state_path.is_file():
        counts = read_state_counts(state_path)
    stat = os.statvfs(root if root.exists() else root.parent)
    result = {
        "staging_root": str(root),
        "manifests": {
            name: (root / "manifests" / name).is_file()
            for name in manifests
        },
        "state_counts": counts,
        "free_bytes": stat.f_bavail * stat.f_frsize,
        "free_inodes": stat.f_favail,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)


def add_execution_options(parser: argparse.ArgumentParser) -> None:
    add_paths(parser)
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    parser.add_argument(
        "--verify-resume-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def add_reference_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment")
    add_execution_options(segment)
    segment.set_defaults(func=run_segment)

    represent = subparsers.add_parser("represent")
    add_execution_options(represent)
    add_reference_option(represent)
    represent.set_defaults(func=run_represent)

    stats = subparsers.add_parser("stats")
    add_execution_options(stats)
    add_reference_option(stats)
    stats.set_defaults(func=run_stats)

    verify = subparsers.add_parser("verify")
    add_paths(verify)
    add_reference_option(verify)
    verify.set_defaults(func=run_verify)

    ready = subparsers.add_parser("ready")
    add_paths(ready)
    add_reference_option(ready)
    ready.set_defaults(func=run_ready)

    status = subparsers.add_parser("status")
    add_paths(status)
    status.set_defaults(func=run_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        args.func(args)
    except (
        PipelineError,
        FileNotFoundError,
        PermissionError,
        sqlite3.Error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
