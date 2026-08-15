#!/usr/bin/env python3
"""Safe, resumable low-disk reconstruction of the HumanML3D pose cache.

The script never writes into HumanML3D-official.  It validates the licensed
archives as streams and materializes only the 22 joints used downstream.
Current stages are preflight, pose, verify-pose and status.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import BinaryIO, Iterable, Mapping, Sequence
import zipfile


SCHEMA_VERSION = 2
ALGORITHM_REVISION = "humanml3d-pose-cache-v2"
TARGET_FPS = 20
OFFICIAL_INDEX_ROWS = 14_616
OFFICIAL_UNIQUE_SOURCES = 11_715
POSE_JOINTS = 22
DATA_ROOT = Path("/root/gpufree-data")
SOURCE_ROOT = DATA_ROOT / "datasets/HumanML3D-official"
ARCHIVE_ROOT = DATA_ROOT / "datasets/humanml3d/amass_archives"
STAGING_ROOT = DATA_ROOT / "datasets/practice9/humanml3d_rebuild/staging-v1"
ARCHIVE_MANIFEST = DATA_ROOT / "tmp/humanml3d/amass/sizes.tsv"

SOURCE_TRIM_FRAMES = {
    "Eyes_Japan_Dataset": 3 * TARGET_FPS,
    "MPI_HDM05": 3 * TARGET_FPS,
    "TotalCapture": TARGET_FPS,
    "MPI_Limits": TARGET_FPS,
    "Transitions_mocap": TARGET_FPS // 2,
}

# Download filename stem -> canonical directory stored in the archive.
ARCHIVE_TOPS: dict[str, str] = {
    "ACCAD": "ACCAD",
    "BMLhandball": "BMLhandball",
    "BMLmovi": "BMLmovi",
    "BMLrub": "BioMotionLab_NTroje",
    "CMU": "CMU",
    "DFaust": "DFaust_67",
    "EKUT": "EKUT",
    "EyesJapanDataset": "Eyes_Japan_Dataset",
    "HDM05": "MPI_HDM05",
    "HumanEva": "HumanEva",
    "KIT": "KIT",
    "MoSh": "MPI_mosh",
    "PosePrior": "MPI_Limits",
    "SFU": "SFU",
    "SSM": "SSM_synced",
    "TCDHands": "TCD_handMocap",
    "TotalCapture": "TotalCapture",
    "Transitions": "Transitions_mocap",
}
TOP_TO_ARCHIVE = {top: archive for archive, top in ARCHIVE_TOPS.items()}
MODEL_FILES = (
    "body_models/smplh/male/model.npz",
    "body_models/smplh/female/model.npz",
    "body_models/dmpls/male/model.npz",
    "body_models/dmpls/female/model.npz",
)


class PipelineError(RuntimeError):
    """A controlled pipeline failure."""


class SafetyError(PipelineError):
    """An input violated a path, type, or integrity boundary."""


@dataclasses.dataclass(frozen=True)
class IndexRow:
    source_rel: str
    start_frame: int
    end_frame: int
    new_name: str


@dataclasses.dataclass(frozen=True)
class IndexContract:
    rows: tuple[IndexRow, ...]
    unique_sources: frozenset[str]
    minimum_frames: Mapping[str, int]
    required_by_archive: Mapping[str, frozenset[str]]
    required_humanact: frozenset[str]


@dataclasses.dataclass(frozen=True)
class ArchiveScan:
    archive: str
    canonical_top: str
    members: int
    regular_files: int
    expanded_bytes: int
    required_members: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class GpuSnapshot:
    index: str
    uuid: str
    memory_used_mib: int
    utilization_percent: int
    compute_pids: frozenset[int]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def json_dump_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


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


def build_algorithm_contract(
    models: Mapping[str, object],
) -> dict[str, object]:
    model_hashes: dict[str, str] = {}
    for relative in MODEL_FILES:
        metadata = models.get(relative)
        if not isinstance(metadata, Mapping):
            raise PipelineError(f"missing model metadata for {relative}")
        model_hashes[relative] = validate_sha256(
            metadata.get("sha256"), label=relative
        )
    payload: dict[str, object] = {
        "revision": ALGORITHM_REVISION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "target_fps": TARGET_FPS,
        "pose_joints": POSE_JOINTS,
        "model_sha256": model_hashes,
    }
    return {**payload, "fingerprint": canonical_sha256(payload)}


def build_pose_fingerprint(
    algorithm_fingerprint: object,
    index_sha256: object,
) -> str:
    return canonical_sha256(
        {
            "algorithm_fingerprint": validate_sha256(
                algorithm_fingerprint,
                label="algorithm fingerprint",
            ),
            "index_sha256": validate_sha256(
                index_sha256,
                label="index.csv",
            ),
        }
    )


def normalize_member_name(name: str) -> str:
    """Normalize a member while retaining strict traversal checks."""
    if not name or "\x00" in name or "\\" in name:
        raise SafetyError(f"unsafe archive member name: {name!r}")
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise SafetyError(f"unsafe archive member name: {name!r}")
    return path.as_posix()


def checked_output_path(root: Path, relative: str) -> Path:
    normalized = normalize_member_name(relative)
    resolved_root = root.resolve()
    candidate = (resolved_root / normalized).resolve()
    if candidate == resolved_root or not candidate.is_relative_to(resolved_root):
        raise SafetyError(f"output escapes staging root: {relative!r}")
    return candidate


def source_path_to_relative(source_path: str) -> str:
    prefix = "./pose_data/"
    if not source_path.startswith(prefix):
        raise SafetyError(f"index source is outside ./pose_data: {source_path!r}")
    return normalize_member_name(source_path[len(prefix) :])


def parse_frame(
    value: str,
    field: str,
    row_number: int,
    *,
    allow_minus_one: bool = False,
) -> int:
    try:
        number = float(value)
        integer = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError(
            f"index row {row_number}: invalid {field}={value!r}"
        ) from exc
    minimum = -1 if allow_minus_one else 0
    if number != integer or integer < minimum:
        raise PipelineError(f"index row {row_number}: invalid {field}={value!r}")
    return integer


def load_index_contract(
    index_path: Path, *, enforce_official_counts: bool = True
) -> IndexContract:
    fields = {"source_path", "start_frame", "end_frame", "new_name"}
    rows: list[IndexRow] = []
    names: set[str] = set()
    sources: set[str] = set()
    minimum_frames: dict[str, int] = {}
    by_archive: dict[str, set[str]] = {name: set() for name in ARCHIVE_TOPS}
    humanact: set[str] = set()
    with index_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not fields.issubset(reader.fieldnames or ()):
            raise PipelineError(f"index is missing columns: {sorted(fields)}")
        for row_number, raw in enumerate(reader, start=2):
            source_rel = source_path_to_relative(raw["source_path"])
            source = PurePosixPath(source_rel)
            top = source.parts[0]
            is_humanact = top == "humanact12"
            new_name = normalize_member_name(raw["new_name"])
            if "/" in new_name or not new_name.endswith(".npy"):
                raise PipelineError(
                    f"index row {row_number}: invalid new_name={new_name!r}"
                )
            if new_name in names:
                raise PipelineError(
                    f"index row {row_number}: duplicate new_name={new_name!r}"
                )
            names.add(new_name)
            start = parse_frame(
                raw["start_frame"],
                "start_frame",
                row_number,
                allow_minus_one=is_humanact,
            )
            end = parse_frame(
                raw["end_frame"],
                "end_frame",
                row_number,
                allow_minus_one=is_humanact,
            )
            if not is_humanact and end < start:
                raise PipelineError(
                    f"index row {row_number}: end_frame precedes start_frame"
                )
            rows.append(IndexRow(source_rel, start, end, new_name))
            sources.add(source_rel)
            if is_humanact:
                if (
                    len(source.parts) < 3
                    or source.parts[:2] != ("humanact12", "humanact12")
                ):
                    raise SafetyError(
                        f"unexpected HumanAct12 index layout: {source_rel}"
                    )
                humanact.add(source_rel)
            else:
                try:
                    archive_name = TOP_TO_ARCHIVE[top]
                except KeyError as exc:
                    raise PipelineError(
                        f"index references unknown AMASS top-level directory: {top}"
                    ) from exc
                by_archive[archive_name].add(
                    source.with_suffix(".npz").as_posix()
                )
            required_frames = (
                1
                if is_humanact
                else SOURCE_TRIM_FRAMES.get(top, 0) + end
            )
            minimum_frames[source_rel] = max(
                minimum_frames.get(source_rel, 1), required_frames, 1
            )
    mirrored = {"M" + name for name in names}
    if names & mirrored:
        raise PipelineError("mirrored output names collide with base names")
    if enforce_official_counts:
        if len(rows) != OFFICIAL_INDEX_ROWS:
            raise PipelineError(
                f"expected {OFFICIAL_INDEX_ROWS} rows, found {len(rows)}"
            )
        if len(sources) != OFFICIAL_UNIQUE_SOURCES:
            raise PipelineError(
                f"expected {OFFICIAL_UNIQUE_SOURCES} sources, found {len(sources)}"
            )
        if len(names | mirrored) != 29_232:
            raise PipelineError("official base+mirror contract is not 29,232")
    return IndexContract(
        rows=tuple(rows),
        unique_sources=frozenset(sources),
        minimum_frames=dict(minimum_frames),
        required_by_archive={
            name: frozenset(value) for name, value in by_archive.items()
        },
        required_humanact=frozenset(humanact),
    )


def load_archive_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise PipelineError(
                    f"archive manifest line {line_number} has fewer than two fields"
                )
            name, value = parts[:2]
            if name in sizes:
                raise PipelineError(f"archive manifest contains duplicate {name}")
            try:
                sizes[name] = int(value)
            except ValueError as exc:
                raise PipelineError(
                    f"archive manifest has invalid size for {name}"
                ) from exc
    if set(sizes) != set(ARCHIVE_TOPS):
        missing = sorted(set(ARCHIVE_TOPS) - set(sizes))
        extra = sorted(set(sizes) - set(ARCHIVE_TOPS))
        raise PipelineError(
            f"archive manifest mismatch; missing={missing}, extra={extra}"
        )
    return sizes


class HashingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data

    def readable(self) -> bool:
        return True


def scan_tar_archive(
    path: Path,
    archive_name: str,
    canonical_top: str,
    required_members: Iterable[str],
) -> ArchiveScan:
    required = {normalize_member_name(item) for item in required_members}
    found: set[str] = set()
    seen: set[str] = set()
    members = regular = expanded = 0
    before = path.stat()
    with path.open("rb") as raw:
        reader = HashingReader(raw)
        try:
            with tarfile.open(fileobj=reader, mode="r|bz2") as archive:
                for member in archive:
                    members += 1
                    name = normalize_member_name(member.name)
                    if name in seen:
                        raise SafetyError(
                            f"{archive_name}: duplicate normalized member {name}"
                        )
                    seen.add(name)
                    if name == "LICENSE.txt":
                        if not member.isreg() or member.size < 0:
                            raise SafetyError(
                                f"{archive_name}: unsafe root LICENSE.txt"
                            )
                        regular += 1
                        expanded += member.size
                        continue
                    if PurePosixPath(name).parts[0] != canonical_top:
                        raise SafetyError(
                            f"{archive_name}: {name!r} is outside "
                            f"{canonical_top!r}"
                        )
                    if member.isdir():
                        continue
                    if not member.isreg() or member.size < 0:
                        raise SafetyError(
                            f"{archive_name}: unsafe member {name!r}"
                        )
                    regular += 1
                    expanded += member.size
                    if name in required:
                        found.add(name)
            while reader.read(4 * 1024 * 1024):
                pass
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise SafetyError(f"cannot safely scan {path}: {exc}") from exc
        digest = reader.digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SafetyError(f"archive changed while scanning: {path}")
    missing = sorted(required - found)
    if missing:
        raise PipelineError(
            f"{archive_name}: {len(missing)} required sources absent; "
            f"first={missing[:5]}"
        )
    return ArchiveScan(
        archive_name,
        canonical_top,
        members,
        regular,
        expanded,
        len(required),
        digest,
    )


def validate_zip_info(info: zipfile.ZipInfo, canonical_top: str) -> str:
    name = normalize_member_name(info.filename)
    if PurePosixPath(name).parts[0] != canonical_top:
        raise SafetyError(f"zip member {name!r} is outside {canonical_top!r}")
    unix_type = (info.external_attr >> 16) & 0o170000
    if info.is_dir():
        if unix_type not in (0, stat.S_IFDIR):
            raise SafetyError(f"unsafe zip directory type: {name}")
    elif unix_type not in (0, stat.S_IFREG):
        raise SafetyError(f"zip member is not regular: {name}")
    return name


def scan_humanact_zip(
    path: Path, required_members: Iterable[str]
) -> dict[str, object]:
    required = {normalize_member_name(item) for item in required_members}
    found: set[str] = set()
    seen: set[str] = set()
    expanded = 0
    before = path.stat()
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise SafetyError(f"HumanAct12 CRC failure at {bad!r}")
            for info in archive.infolist():
                name = validate_zip_info(info, "humanact12")
                if name in seen:
                    raise SafetyError(
                        f"HumanAct12 duplicate normalized member: {name}"
                    )
                seen.add(name)
                if not info.is_dir():
                    expanded += info.file_size
                    if name in required:
                        found.add(name)
    except (zipfile.BadZipFile, OSError) as exc:
        raise SafetyError(f"cannot safely scan {path}: {exc}") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SafetyError(f"HumanAct12 changed while scanning: {path}")
    missing = sorted(required - found)
    if missing:
        raise PipelineError(
            f"HumanAct12 is missing {len(missing)} sources; first={missing[:5]}"
        )
    return {
        "members": len(seen),
        "required_members": len(required),
        "expanded_bytes": expanded,
        "sha256": sha256_file(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def find_archive_users(root: Path) -> list[dict[str, object]]:
    needle = str(root.resolve()).encode()
    matches: list[dict[str, object]] = []
    competing_tools = {
        "bzip2",
        "curl",
        "download_humanml3d_amass.sh",
        "tar",
    }
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            command = (proc / "cmdline").read_bytes()
        except (OSError, ValueError):
            continue
        if pid == os.getpid() or needle not in command:
            continue
        tokens = [
            item.decode(errors="replace")
            for item in command.split(b"\0")
            if item
        ]
        tools = {Path(item).name for item in tokens}
        if not tools & competing_tools:
            continue
        matches.append({"pid": pid, "tools": sorted(tools & competing_tools)})
    return sorted(matches, key=lambda item: int(item["pid"]))


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream: io.TextIOWrapper | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise PipelineError(f"another rebuild owns {self.path}") from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def import_numpy():
    import numpy as np

    return np


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
                f"{label}: disk reserve gate failed; "
                f"free={free}, required={required}"
            )


def validate_pose_array(
    array: object,
    *,
    label: str,
    minimum_frames: int = 1,
) -> None:
    np = import_numpy()
    if minimum_frames < 1:
        raise PipelineError(f"{label}: invalid minimum_frames")
    if not isinstance(array, np.ndarray):
        raise PipelineError(f"{label}: expected ndarray")
    if (
        array.ndim != 3
        or array.shape[0] < minimum_frames
        or array.shape[1] < POSE_JOINTS
        or array.shape[2] != 3
    ):
        raise PipelineError(f"{label}: invalid pose shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise PipelineError(f"{label}: non-numeric dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise PipelineError(f"{label}: contains NaN or Inf")


def validate_pose_file(
    path: Path,
    *,
    minimum_frames: int = 1,
) -> tuple[tuple[int, ...], str]:
    np = import_numpy()
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PipelineError(f"cannot load pose cache {path}: {exc}") from exc
    validate_pose_array(
        array,
        label=str(path),
        minimum_frames=minimum_frames,
    )
    return tuple(int(item) for item in array.shape), str(array.dtype)


def save_npy_atomic(
    path: Path,
    array: object,
    *,
    space_guard: SpaceGuard | None = None,
) -> None:
    np = import_numpy()
    validate_pose_array(array, label=str(path))
    if space_guard is not None:
        space_guard.require(
            int(array.nbytes) + 4096,
            label=f"write {path}",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
        if space_guard is not None:
            space_guard.require(label=f"post-write {path}")
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS items(
                stage TEXT NOT NULL, item_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL, output_rel TEXT NOT NULL,
                output_size INTEGER NOT NULL, output_sha256 TEXT NOT NULL,
                shape_json TEXT NOT NULL, dtype TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY(stage,item_key))
            """
        )
        self.db.commit()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.db.close()

    def complete(
        self,
        stage: str,
        key: str,
        fingerprint: str,
        output: Path,
        *,
        verify_hash: bool,
        minimum_frames: int = 1,
    ) -> bool:
        row = self.db.execute(
            "SELECT fingerprint,output_size,output_sha256 FROM items "
            "WHERE stage=? AND item_key=?",
            (stage, key),
        ).fetchone()
        if row is None or row[0] != fingerprint or not output.is_file():
            return False
        if output.stat().st_size != row[1]:
            return False
        try:
            validate_pose_file(
                output,
                minimum_frames=minimum_frames,
            )
        except PipelineError:
            return False
        return not verify_hash or sha256_file(output) == row[2]

    def record(
        self,
        stage: str,
        key: str,
        fingerprint: str,
        output_rel: str,
        output: Path,
        shape: Sequence[int],
        dtype: str,
    ) -> None:
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
                output_rel,
                output.stat().st_size,
                sha256_file(output),
                json.dumps(list(shape)),
                dtype,
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


def force_cpu_only_environment() -> None:
    if "torch" in sys.modules:
        raise PipelineError("torch imported before CPU device guard")
    sys.dont_write_bytecode = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def parse_nspid_status(
    status_text: str,
    fallback_pid: int,
) -> frozenset[int]:
    """Return current process IDs across nested PID namespaces."""
    identifiers = {fallback_pid}
    for line in status_text.splitlines():
        key, separator, values = line.partition(":")
        if separator and key.strip() == "NSpid":
            for token in values.split():
                try:
                    identifier = int(token)
                except ValueError:
                    continue
                if identifier > 0:
                    identifiers.add(identifier)
            break
    return frozenset(identifiers)


def current_process_ids(
    status_path: Path = Path("/proc/self/status"),
    *,
    fallback_pid: int | None = None,
) -> frozenset[int]:
    fallback = os.getpid() if fallback_pid is None else fallback_pid
    try:
        status_text = status_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return frozenset({fallback})
    return parse_nspid_status(status_text, fallback)


def parse_gpu_snapshots(
    gpu_rows: str, compute_rows: str
) -> dict[str, GpuSnapshot]:
    base: dict[str, tuple[str, int, int]] = {}
    for line in gpu_rows.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise PipelineError(f"unexpected nvidia-smi row: {line!r}")
        base[fields[1]] = (fields[0], int(fields[2]), int(fields[3]))
    pids: dict[str, set[int]] = {uuid: set() for uuid in base}
    for line in compute_rows.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 2 or fields[0] not in pids:
            raise PipelineError(f"unexpected compute row: {line!r}")
        pids[fields[0]].add(int(fields[1]))
    return {
        uuid: GpuSnapshot(index, uuid, memory, util, frozenset(pids[uuid]))
        for uuid, (index, memory, util) in base.items()
    }


def query_gpu_snapshots() -> dict[str, GpuSnapshot]:
    def query(kind: str, fields: str) -> str:
        try:
            return subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-{kind}={fields}",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PipelineError(f"nvidia-smi query failed: {exc}") from exc

    gpu = query("gpu", "index,uuid,memory.used,utilization.gpu")
    apps = query("compute-apps", "gpu_uuid,pid,used_gpu_memory")
    return parse_gpu_snapshots(gpu, apps)


class GpuLease:
    def __init__(
        self,
        index: str | None,
        uuid: str | None,
        data_root: Path,
        max_memory: int,
        max_util: int,
    ):
        self.index = index
        self.requested_uuid = uuid
        self.data_root = data_root
        self.max_memory = max_memory
        self.max_util = max_util
        self.snapshot: GpuSnapshot | None = None
        self.lock: FileLock | None = None
        self.baseline_compute_pids: frozenset[int] | None = None
        self.owned_compute_pid: int | None = None

    def __enter__(self) -> "GpuLease":
        if "torch" in sys.modules:
            raise PipelineError("torch imported before CUDA visibility guard")
        if (self.index is None) == (self.requested_uuid is None):
            raise PipelineError("select exactly one of GPU index or UUID")
        snapshots = query_gpu_snapshots()
        if self.requested_uuid is not None:
            candidate = snapshots.get(self.requested_uuid)
            candidates = [] if candidate is None else [candidate]
            selector = f"UUID {self.requested_uuid}"
        else:
            candidates = [
                item
                for item in snapshots.values()
                if item.index == self.index
            ]
            selector = f"index {self.index}"
        if len(candidates) != 1:
            raise PipelineError(f"physical GPU {selector} does not exist")
        self.snapshot = candidates[0]
        self.index = self.snapshot.index
        if (
            self.snapshot.compute_pids
            or self.snapshot.memory_used_mib > self.max_memory
            or self.snapshot.utilization_percent > self.max_util
        ):
            raise PipelineError(f"physical GPU {self.index} is not idle")
        safe_uuid = self.snapshot.uuid.replace("/", "_")
        self.lock = FileLock(
            self.data_root / f"tmp/humanml3d/gpu-{safe_uuid}.lock"
        )
        self.lock.__enter__()
        second = query_gpu_snapshots().get(self.snapshot.uuid)
        if (
            second is None
            or second.index != self.snapshot.index
            or second.compute_pids
            or second.memory_used_mib > self.max_memory
            or second.utilization_percent > self.max_util
        ):
            self.lock.__exit__(None, None, None)
            self.lock = None
            raise PipelineError(f"physical GPU {self.index} became busy")
        self.baseline_compute_pids = frozenset(
            set(self.snapshot.compute_pids) | set(second.compute_pids)
        )
        self.snapshot = second
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.snapshot.uuid
        return self

    @property
    def uuid(self) -> str:
        if self.snapshot is None:
            raise PipelineError("GPU lease has not been acquired")
        return self.snapshot.uuid

    def bind_cuda_process(self, torch_module: object) -> None:
        if (
            self.snapshot is None
            or self.lock is None
            or self.baseline_compute_pids is None
        ):
            raise PipelineError("GPU lease was not acquired")
        if self.baseline_compute_pids:
            raise PipelineError("GPU baseline was not empty")
        if torch_module.cuda.device_count() != 1:
            raise PipelineError("CUDA binding requires exactly one visible GPU")
        try:
            probe = torch_module.empty(1, device="cuda:0")
            torch_module.cuda.synchronize()
        except Exception as exc:
            raise PipelineError("CUDA context probe failed") from exc
        time.sleep(0.2)
        first = query_gpu_snapshots().get(self.snapshot.uuid)
        time.sleep(0.2)
        second = query_gpu_snapshots().get(self.snapshot.uuid)
        if (
            first is None
            or second is None
            or first.index != self.snapshot.index
            or second.index != self.snapshot.index
        ):
            raise PipelineError("GPU changed during CUDA process binding")
        first_pids = set(first.compute_pids)
        second_pids = set(second.compute_pids)
        if first_pids != second_pids or len(first_pids) != 1:
            raise PipelineError(
                "GPU process set was not one stable PID during CUDA bind: "
                f"{sorted(first_pids)} -> {sorted(second_pids)}"
            )
        self.owned_compute_pid = next(iter(first_pids))
        self_ids = set(current_process_ids())
        print(
            f"GPU_LEASE_BOUND uuid={self.snapshot.uuid} "
            f"owned_compute_pid={self.owned_compute_pid} "
            f"namespace_visible={self.owned_compute_pid in self_ids}",
            flush=True,
        )
        del probe


    def assert_no_foreign_compute(self) -> None:
        if self.snapshot is None:
            return
        current = query_gpu_snapshots().get(self.snapshot.uuid)
        if current is None or current.index != self.snapshot.index:
            raise PipelineError(
                f"GPU {self.snapshot.uuid} disappeared or changed index"
            )
        if self.owned_compute_pid is None:
            raise PipelineError("GPU compute process was not bound")
        observed = set(current.compute_pids)
        expected = {self.owned_compute_pid}
        if observed != expected:
            raise PipelineError(
                f"GPU {self.index} process set changed: "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )

    def __exit__(self, *_: object) -> None:
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None


def load_amass_record(
    stream: BinaryIO,
    label: str,
) -> tuple[int, object, object, object, str]:
    np = import_numpy()
    stream.seek(0)
    try:
        with np.load(stream, allow_pickle=False) as data:
            required = {
                "mocap_framerate",
                "trans",
                "poses",
                "gender",
                "betas",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise PipelineError(f"{label}: missing keys {missing}")
            fps_value = np.asarray(data["mocap_framerate"])
            poses = np.asarray(data["poses"])
            translations = np.asarray(data["trans"])
            betas = np.asarray(data["betas"])
            gender_value = np.asarray(data["gender"])
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"{label}: invalid AMASS npz: {exc}") from exc
    if (
        fps_value.size != 1
        or not np.issubdtype(fps_value.dtype, np.number)
        or not np.isfinite(fps_value).all()
    ):
        raise PipelineError(f"{label}: invalid mocap_framerate schema")
    fps = float(fps_value.reshape(-1)[0])
    if not np.isfinite(fps) or fps < TARGET_FPS:
        raise PipelineError(f"{label}: invalid fps {fps}")
    stride = int(fps / TARGET_FPS)
    if stride < 1:
        raise PipelineError(f"{label}: invalid stride {stride}")
    if (
        poses.ndim != 2
        or poses.shape[0] < 1
        or poses.shape[1] != 156
        or not np.issubdtype(poses.dtype, np.floating)
        or not np.isfinite(poses).all()
    ):
        raise PipelineError(f"{label}: invalid poses schema {poses.shape}")
    if (
        translations.shape != (poses.shape[0], 3)
        or not np.issubdtype(translations.dtype, np.floating)
        or not np.isfinite(translations).all()
    ):
        raise PipelineError(f"{label}: invalid trans schema {translations.shape}")
    if (
        betas.ndim != 1
        or betas.size < 10
        or not np.issubdtype(betas.dtype, np.floating)
        or not np.isfinite(betas).all()
    ):
        raise PipelineError(f"{label}: invalid betas schema {betas.shape}")
    if gender_value.size != 1 or gender_value.dtype.kind not in ("S", "U"):
        raise PipelineError(f"{label}: invalid gender schema")
    scalar = gender_value.reshape(-1)[0]
    try:
        gender = (
            scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)
        ).strip().lower()
    except UnicodeDecodeError as exc:
        raise PipelineError(f"{label}: invalid gender encoding") from exc
    if gender not in ("male", "female"):
        raise PipelineError(f"{label}: unsupported AMASS gender {gender!r}")
    return stride, poses, translations, betas[:10], gender


class BodyModelRunner:
    def __init__(
        self,
        source_root: Path,
        device_name: str,
        batch_frames: int,
        lease: GpuLease | None,
    ):
        if batch_frames < 1:
            raise PipelineError("batch_frames must be positive")
        self.source_root = source_root
        self.device_name = device_name
        self.batch_frames = batch_frames
        self.lease = lease
        sys.path.insert(0, str(source_root))
        import torch
        from human_body_prior.body_model.body_model import BodyModel

        self.torch = torch
        self.BodyModel = BodyModel
        if device_name == "cpu":
            if torch.cuda.is_available():
                raise PipelineError("CPU stage can see CUDA")
            self.device = torch.device("cpu")
        else:
            if torch.cuda.device_count() != 1:
                raise PipelineError(
                    f"GPU stage sees {torch.cuda.device_count()} CUDA devices"
                )
            self.device = torch.device("cuda:0")
            if lease is None:
                raise PipelineError("CUDA stage requires an acquired GPU lease")
            lease.bind_cuda_process(torch)
        self.models: dict[str, object] = {}

    def model(self, gender: str):
        if gender not in ("male", "female"):
            raise PipelineError(f"unsupported AMASS gender {gender!r}")
        if gender not in self.models:
            self.models[gender] = self.BodyModel(
                bm_fname=str(
                    self.source_root
                    / f"body_models/smplh/{gender}/model.npz"
                ),
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=str(
                    self.source_root
                    / f"body_models/dmpls/{gender}/model.npz"
                ),
            ).to(self.device)
        return self.models[gender]

    def process(self, stream: BinaryIO, label: str):
        np = import_numpy()
        torch = self.torch
        stride, poses, translations, betas, gender = load_amass_record(
            stream, label
        )
        indices = np.arange(0, len(poses), stride, dtype=np.int64)
        output = np.empty((len(indices), POSE_JOINTS, 3), np.float32)
        transform = np.asarray(
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32
        )
        model = self.model(gender)
        offset = 0
        batch = self.batch_frames
        while offset < len(indices):
            end = min(offset + batch, len(indices))
            selected = indices[offset:end]
            pose = poses[selected]
            params = {
                "root_orient": torch.as_tensor(
                    pose[:, :3], dtype=torch.float32, device=self.device
                ),
                "pose_body": torch.as_tensor(
                    pose[:, 3:66], dtype=torch.float32, device=self.device
                ),
                "pose_hand": torch.as_tensor(
                    pose[:, 66:], dtype=torch.float32, device=self.device
                ),
                "trans": torch.as_tensor(
                    translations[selected],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "betas": torch.as_tensor(
                    np.repeat(betas[None], len(selected), axis=0),
                    dtype=torch.float32,
                    device=self.device,
                ),
            }
            try:
                with torch.inference_mode():
                    body = model(**params)
                    joints = (
                        body.Jtr[:, :POSE_JOINTS].detach().cpu().numpy()
                    )
                output[offset:end] = np.matmul(joints, transform)
                offset = end
            except torch.cuda.OutOfMemoryError as exc:
                if self.device_name == "cpu" or batch <= 32:
                    raise PipelineError(
                        f"{label}: OOM at batch size {batch}"
                    ) from exc
                batch = max(32, batch // 2)
                torch.cuda.empty_cache()
        validate_pose_array(output, label=label)
        if self.lease is not None:
            self.lease.assert_no_foreign_compute()
        return output


def copy_stream(source: BinaryIO, target: BinaryIO, expected: int) -> None:
    copied = 0
    while chunk := source.read(4 * 1024 * 1024):
        copied += len(chunk)
        if copied > expected:
            raise SafetyError("archive member exceeded declared size")
        target.write(chunk)
    if copied != expected:
        raise SafetyError(
            f"archive member size mismatch: expected {expected}, got {copied}"
        )
    target.seek(0)


def materialize_humanact(
    zip_path: Path,
    required_members: Iterable[str],
    pose_root: Path,
    state: StateDB,
    input_fingerprint: str,
    minimum_frames: Mapping[str, int],
    space_guard: SpaceGuard,
    *,
    verify_resume_hash: bool,
) -> int:
    np = import_numpy()
    required = {normalize_member_name(item) for item in required_members}
    completed = 0
    with zipfile.ZipFile(zip_path) as archive:
        infos = {
            validate_zip_info(info, "humanact12"): info
            for info in archive.infolist()
        }
        missing = sorted(required - set(infos))
        if missing:
            raise PipelineError(f"HumanAct12 lost files {missing[:5]}")
        for name in sorted(required):
            info = infos[name]
            output = checked_output_path(pose_root, name)
            minimum = minimum_frames[name]
            space_guard.require(label=f"HumanAct12 item {name}")
            fingerprint = (
                f"{input_fingerprint}:{info.CRC}:{info.file_size}"
            )
            if state.complete(
                "pose",
                name,
                fingerprint,
                output,
                verify_hash=verify_resume_hash,
                minimum_frames=minimum,
            ):
                completed += 1
                continue
            with archive.open(info) as stream:
                raw = stream.read(info.file_size + 1)
            if len(raw) != info.file_size:
                raise SafetyError(f"HumanAct12 size mismatch: {name}")
            try:
                array = np.load(io.BytesIO(raw), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise PipelineError(
                    f"cannot load HumanAct12 member {name}: {exc}"
                ) from exc
            save_npy_atomic(output, array, space_guard=space_guard)
            shape, dtype = validate_pose_file(
                output,
                minimum_frames=minimum,
            )
            state.record(
                "pose", name, fingerprint, name, output, shape, dtype
            )
            completed += 1
    return completed


def existing_device(path: Path) -> int:
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise SafetyError(f"cannot resolve device for {path}")
        current = parent
    return current.stat().st_dev


def validate_layout(
    source_root: Path,
    archive_root: Path,
    staging_root: Path,
    data_root: Path,
) -> None:
    data = data_root.resolve()
    trusted_data = DATA_ROOT.resolve()
    if data != trusted_data:
        raise SafetyError(
            f"data root must be pinned to {trusted_data}, got {data}"
        )
    if not data.is_dir():
        raise SafetyError(f"data disk is unavailable: {data}")
    source = source_root.resolve()
    archives = archive_root.resolve()
    staging = staging_root.resolve()
    if not source.is_dir() or not archives.is_dir():
        raise SafetyError("source and archive roots must exist as directories")
    data_device = data.stat().st_dev
    for label, path in (
        ("source", source),
        ("archives", archives),
        ("staging", staging),
    ):
        if not path.is_relative_to(data):
            raise SafetyError(f"{label} path outside data disk: {path}")
        if existing_device(path) != data_device:
            raise SafetyError(
                f"{label} path is not on the data disk device: {path}"
            )
    if (
        staging == source
        or staging.is_relative_to(source)
        or source.is_relative_to(staging)
    ):
        raise SafetyError("staging and HumanML3D-official must be disjoint")
    if (
        staging == archives
        or staging.is_relative_to(archives)
        or archives.is_relative_to(staging)
    ):
        raise SafetyError("staging and archive root must be disjoint")


def configure_runtime_environment(staging_root: Path) -> dict[str, Path]:
    runtime_root = staging_root.resolve() / "runtime"
    paths = {
        "TMPDIR": runtime_root / "tmp",
        "TEMP": runtime_root / "tmp",
        "TMP": runtime_root / "tmp",
        "XDG_CACHE_HOME": runtime_root / "xdg-cache",
        "TORCH_HOME": runtime_root / "torch-cache",
        "CUDA_CACHE_PATH": runtime_root / "cuda-cache",
    }
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(paths["TMPDIR"])
    return paths


def check_models(source_root: Path, initialize: bool) -> dict[str, object]:
    result: dict[str, object] = {}
    for relative in MODEL_FILES:
        path = source_root / relative
        if not path.is_file() or path.stat().st_size < 1024:
            raise PipelineError(f"missing body model: {path}")
        result[relative] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
    if initialize:
        force_cpu_only_environment()
        sys.path.insert(0, str(source_root))
        import torch
        from human_body_prior.body_model.body_model import BodyModel

        if torch.cuda.is_available():
            raise PipelineError("model preflight unexpectedly sees CUDA")
        for gender in ("male", "female"):
            model = BodyModel(
                bm_fname=str(
                    source_root / f"body_models/smplh/{gender}/model.npz"
                ),
                num_betas=10,
                num_dmpls=8,
                dmpl_fname=str(
                    source_root / f"body_models/dmpls/{gender}/model.npz"
                ),
            ).to(torch.device("cpu"))
            if tuple(model.f.shape) != (13_776, 3):
                raise PipelineError(
                    f"unexpected {gender} faces shape {tuple(model.f.shape)}"
                )
            del model
    return result


def archive_metadata(path: Path, expected_size: int) -> dict[str, int]:
    if not path.is_file():
        raise PipelineError(f"missing archive: {path}")
    info = path.stat()
    if info.st_size != expected_size:
        raise PipelineError(
            f"{path.name}: expected {expected_size}, got {info.st_size}"
        )
    with path.open("rb") as stream:
        if stream.read(3) != b"BZh":
            raise SafetyError(f"invalid bzip2 signature: {path}")
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def run_preflight(args: argparse.Namespace) -> None:
    validate_layout(
        args.source_root, args.archive_root, args.staging_root, args.data_root
    )
    configure_runtime_environment(args.staging_root)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    with FileLock(args.staging_root / ".pipeline.lock"):
        users = find_archive_users(args.archive_root)
        if users and not args.allow_concurrent_archive_readers:
            raise PipelineError(f"archive directory already in use: {users}")
        contract = load_index_contract(args.source_root / "index.csv")
        index_sha256 = sha256_file(args.source_root / "index.csv")
        sizes = load_archive_sizes(args.archive_manifest)
        free = shutil.disk_usage(args.staging_root).free
        required = int(
            (args.reserve_gib + args.projected_output_gib) * 1024**3
        )
        if free < required:
            raise PipelineError(
                f"insufficient disk: {free / 1024**3:.2f} GiB free, "
                f"{required / 1024**3:.2f} GiB required"
            )
        scans: dict[str, object] = {}
        archive_stats: dict[str, object] = {}
        for name, top in ARCHIVE_TOPS.items():
            path = args.archive_root / f"{name}.tar.bz2"
            archive_stats[name] = archive_metadata(path, sizes[name])
            scan = scan_tar_archive(
                path, name, top, contract.required_by_archive[name]
            )
            scans[name] = dataclasses.asdict(scan)
            print(
                f"SCANNED {name}: required={scan.required_members} "
                f"files={scan.regular_files}",
                flush=True,
            )
        humanact_path = args.source_root / "pose_data/humanact12.zip"
        humanact = scan_humanact_zip(
            humanact_path, contract.required_humanact
        )
        models = check_models(
            args.source_root, initialize=not args.skip_model_init
        )
        algorithm = build_algorithm_contract(models)
        pose_fingerprint = build_pose_fingerprint(
            algorithm["fingerprint"], index_sha256
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "source_root": str(args.source_root.resolve()),
            "archive_root": str(args.archive_root.resolve()),
            "staging_root": str(args.staging_root.resolve()),
            "index": {
                "sha256": index_sha256,
                "rows": len(contract.rows),
                "unique_sources": len(contract.unique_sources),
                "humanact_sources": len(contract.required_humanact),
            },
            "archive_stats": archive_stats,
            "archives": scans,
            "humanact": humanact,
            "models": models,
            "algorithm": algorithm,
            "pose_fingerprint": pose_fingerprint,
            "disk": {
                "free_bytes": free,
                "reserve_gib": args.reserve_gib,
                "projected_output_gib": args.projected_output_gib,
            },
        }
        output = args.staging_root / "manifests/preflight.json"
        json_dump_atomic(output, payload)
        print(f"PREFLIGHT_READY {output}")


def load_preflight(path: Path, args: argparse.Namespace) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot load preflight manifest: {exc}") from exc
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError("preflight schema mismatch")
    for key, expected in (
        ("source_root", str(args.source_root.resolve())),
        ("archive_root", str(args.archive_root.resolve())),
        ("staging_root", str(args.staging_root.resolve())),
    ):
        if value.get(key) != expected:
            raise PipelineError(f"preflight {key} mismatch")
    index = value.get("index")
    if not isinstance(index, Mapping):
        raise PipelineError("preflight index metadata missing")
    current_index_sha256 = sha256_file(args.source_root / "index.csv")
    if index.get("sha256") != current_index_sha256:
        raise PipelineError("index.csv changed after preflight")
    current_models = check_models(args.source_root, initialize=False)
    current_algorithm = build_algorithm_contract(current_models)
    if value.get("algorithm") != current_algorithm:
        raise PipelineError("algorithm or body models changed after preflight")
    expected_pose_fingerprint = build_pose_fingerprint(
        current_algorithm["fingerprint"],
        current_index_sha256,
    )
    if value.get("pose_fingerprint") != expected_pose_fingerprint:
        raise PipelineError("preflight pose fingerprint mismatch")
    disk = value.get("disk")
    if not isinstance(disk, Mapping):
        raise PipelineError("preflight disk metadata missing")
    try:
        reserve_gib = float(disk["reserve_gib"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PipelineError("preflight disk reserve is invalid") from exc
    if reserve_gib != reserve_gib or not 0 <= reserve_gib < 10_000:
        raise PipelineError("preflight disk reserve is invalid")
    return value


def process_archive(
    name: str,
    path: Path,
    required_members: Iterable[str],
    archive_sha256: str,
    pose_fingerprint: str,
    minimum_frames: Mapping[str, int],
    space_guard: SpaceGuard,
    pose_root: Path,
    temporary_root: Path,
    state: StateDB,
    runner: BodyModelRunner,
    verify_resume_hash: bool,
) -> int:
    required = {normalize_member_name(item) for item in required_members}
    found: set[str] = set()
    seen: set[str] = set()
    completed = 0
    before = path.stat()
    canonical_top = ARCHIVE_TOPS[name]
    with path.open("rb") as raw:
        reader = HashingReader(raw)
        try:
            with tarfile.open(fileobj=reader, mode="r|bz2") as archive:
                for member in archive:
                    member_name = normalize_member_name(member.name)
                    if member_name in seen:
                        raise SafetyError(
                            f"{name}: duplicate member {member_name}"
                        )
                    seen.add(member_name)
                    if member_name == "LICENSE.txt":
                        if not member.isreg() or member.size < 0:
                            raise SafetyError(
                                f"{name}: unsafe root LICENSE.txt"
                            )
                        continue
                    if PurePosixPath(member_name).parts[0] != canonical_top:
                        raise SafetyError(
                            f"{name}: {member_name!r} outside "
                            f"{canonical_top!r}"
                        )
                    if member.isdir():
                        continue
                    if not member.isreg() or member.size < 0:
                        raise SafetyError(
                            f"{name}: unsafe member {member_name!r}"
                        )
                    if member_name not in required:
                        continue
                    found.add(member_name)
                    output_rel = (
                        PurePosixPath(member_name)
                        .with_suffix(".npy")
                        .as_posix()
                    )
                    output = checked_output_path(pose_root, output_rel)
                    minimum = minimum_frames[output_rel]
                    space_guard.require(
                        member.size,
                        label=f"spool {name}:{member_name}",
                    )
                    fingerprint = (
                        f"{pose_fingerprint}:{archive_sha256}:"
                        f"{member_name}:{member.size}"
                    )
                    if state.complete(
                        "pose",
                        output_rel,
                        fingerprint,
                        output,
                        verify_hash=verify_resume_hash,
                        minimum_frames=minimum,
                    ):
                        completed += 1
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise SafetyError(f"cannot stream {member_name}")
                    temporary_root.mkdir(parents=True, exist_ok=True)
                    with source, tempfile.SpooledTemporaryFile(
                        max_size=128 * 1024 * 1024,
                        mode="w+b",
                        dir=temporary_root,
                    ) as spool:
                        copy_stream(source, spool, member.size)
                        array = runner.process(
                            spool, f"{name}:{member_name}"
                        )
                    save_npy_atomic(output, array, space_guard=space_guard)
                    shape, dtype = validate_pose_file(
                        output, minimum_frames=minimum
                    )
                    state.record(
                        "pose",
                        output_rel,
                        fingerprint,
                        output_rel,
                        output,
                        shape,
                        dtype,
                    )
                    completed += 1
                    print(
                        f"POSE {name} {completed}/{len(required)} "
                        f"{output_rel}",
                        flush=True,
                    )
            while reader.read(4 * 1024 * 1024):
                pass
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise SafetyError(
                f"cannot safely process archive {path}: {exc}"
            ) from exc
        observed_sha256 = reader.digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SafetyError(f"archive changed during pose stage: {path}")
    if observed_sha256 != archive_sha256:
        raise SafetyError(
            f"archive digest changed after preflight: {path}"
        )
    missing = sorted(required - found)
    if missing:
        raise PipelineError(
            f"{name}: required members disappeared: {missing[:5]}"
        )
    return completed


def verify_pose_tree(
    contract: IndexContract,
    pose_root: Path,
    state: StateDB,
    pose_fingerprint: str,
    index_sha256: str,
    device: Mapping[str, object],
    *,
    verify_hash: bool,
) -> dict[str, object]:
    if not pose_root.is_dir():
        raise PipelineError(f"pose root is missing: {pose_root}")
    expected_sources = set(contract.unique_sources)
    observed_sources: set[str] = set()
    unsafe_paths: list[str] = []
    for candidate in pose_root.rglob("*"):
        relative = candidate.relative_to(pose_root).as_posix()
        if candidate.is_symlink():
            unsafe_paths.append(relative)
        elif candidate.is_file():
            observed_sources.add(relative)
    missing_sources = sorted(expected_sources - observed_sources)
    extra_sources = sorted(observed_sources - expected_sources)
    records = state.records("pose")
    record_keys = set(records)
    output_rels = {
        str(record["output_rel"])
        for record in records.values()
    }
    missing_records = sorted(expected_sources - record_keys)
    extra_records = sorted(record_keys - expected_sources)
    if (
        unsafe_paths
        or missing_sources
        or extra_sources
        or missing_records
        or extra_records
        or output_rels != expected_sources
    ):
        raise PipelineError(
            "pose set mismatch: "
            f"unsafe={unsafe_paths[:5]}, missing={missing_sources[:5]}, "
            f"extra={extra_sources[:5]}, db_missing={missing_records[:5]}, "
            f"db_extra={extra_records[:5]}"
        )
    missing: list[str] = []
    invalid: list[str] = []
    frames = 0
    ledger: list[dict[str, object]] = []
    for relative in sorted(contract.unique_sources):
        path = checked_output_path(pose_root, relative)
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            record = records[relative]
            fingerprint = record["fingerprint"]
            if (
                not isinstance(fingerprint, str)
                or not fingerprint.startswith(f"{pose_fingerprint}:")
                or record["output_rel"] != relative
            ):
                raise PipelineError("DB fingerprint/output binding mismatch")
            minimum = contract.minimum_frames[relative]
            if not state.complete(
                "pose",
                relative,
                fingerprint,
                path,
                verify_hash=verify_hash,
                minimum_frames=minimum,
            ):
                raise PipelineError("DB output integrity mismatch")
            shape, dtype = validate_pose_file(
                path,
                minimum_frames=minimum,
            )
            if record["shape"] != list(shape) or record["dtype"] != dtype:
                raise PipelineError("DB shape/dtype mismatch")
            frames += shape[0]
            ledger.append(
                {
                    "source": relative,
                    "fingerprint": fingerprint,
                    "output_sha256": record["output_sha256"],
                }
            )
        except PipelineError as exc:
            invalid.append(f"{relative}: {exc}")
    if missing or invalid:
        raise PipelineError(
            f"pose verify failed: missing={len(missing)} {missing[:5]}, "
            f"invalid={len(invalid)} {invalid[:5]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "verified_at": utc_now(),
        "sources": len(contract.unique_sources),
        "total_frames": frames,
        "db_records": len(records),
        "pose_fingerprint": pose_fingerprint,
        "index_sha256": index_sha256,
        "record_set_sha256": canonical_sha256(ledger),
        "device": dict(device),
        "finite": True,
        "joint_floor": POSE_JOINTS,
    }


def run_pose(args: argparse.Namespace) -> None:
    validate_layout(
        args.source_root, args.archive_root, args.staging_root, args.data_root
    )
    configure_runtime_environment(args.staging_root)
    preflight = load_preflight(args.preflight_manifest, args)
    pose_fingerprint = validate_sha256(
        preflight["pose_fingerprint"], label="pose fingerprint"
    )
    index_sha256 = validate_sha256(
        preflight["index"]["sha256"], label="index.csv"
    )
    reserve_bytes = int(float(preflight["disk"]["reserve_gib"]) * 1024**3)
    space_guard = SpaceGuard(args.staging_root, reserve_bytes)
    space_guard.require(label="pose startup")
    contract = load_index_contract(args.source_root / "index.csv")
    if preflight["index"]["sha256"] != sha256_file(
        args.source_root / "index.csv"
    ):
        raise PipelineError("index.csv changed after preflight")
    selected = set(args.only_archive or ())
    if not selected:
        selected = set(ARCHIVE_TOPS) | {"HumanAct12"}
    full_run = selected == set(ARCHIVE_TOPS) | {"HumanAct12"}
    with contextlib.ExitStack() as stack:
        stack.enter_context(FileLock(args.staging_root / ".pipeline.lock"))
        lease: GpuLease | None = None
        device_record: dict[str, object] = {"type": "cpu"}
        if args.device == "cpu":
            force_cpu_only_environment()
        else:
            lease = stack.enter_context(
                GpuLease(
                    None if args.gpu_id is None else str(args.gpu_id),
                    args.gpu_uuid,
                    args.data_root,
                    args.max_initial_gpu_memory_mib,
                    args.max_initial_gpu_util,
                )
            )
            device_record = {
                "type": "cuda",
                "physical_index": lease.snapshot.index,
                "uuid": lease.uuid,
            }
        with StateDB(args.staging_root / "state.sqlite3") as state:
            runner: BodyModelRunner | None = None
            for name in ARCHIVE_TOPS:
                if name not in selected:
                    continue
                archive_path = args.archive_root / f"{name}.tar.bz2"
                expected = preflight["archive_stats"][name]
                current = archive_path.stat()
                if (current.st_size, current.st_mtime_ns) != (
                    expected["size"],
                    expected["mtime_ns"],
                ):
                    raise PipelineError(f"{name} changed after preflight")
                if runner is None:
                    runner = BodyModelRunner(
                        args.source_root,
                        args.device,
                        args.batch_frames,
                        lease,
                    )
                process_archive(
                    name,
                    archive_path,
                    contract.required_by_archive[name],
                    preflight["archives"][name]["sha256"],
                    pose_fingerprint,
                    contract.minimum_frames,
                    space_guard,
                    args.staging_root / "pose_data",
                    args.staging_root / "tmp",
                    state,
                    runner,
                    args.verify_resume_hash,
                )
            if "HumanAct12" in selected:
                source = args.source_root / "pose_data/humanact12.zip"
                expected = preflight["humanact"]
                current = source.stat()
                if (current.st_size, current.st_mtime_ns) != (
                    expected["size"],
                    expected["mtime_ns"],
                ):
                    raise PipelineError("HumanAct12 changed after preflight")
                count = materialize_humanact(
                    source,
                    contract.required_humanact,
                    args.staging_root / "pose_data",
                    state,
                    f"{pose_fingerprint}:{expected['sha256']}",
                    contract.minimum_frames,
                    space_guard,
                    verify_resume_hash=args.verify_resume_hash,
                )
                print(
                    f"POSE HumanAct12 {count}/"
                    f"{len(contract.required_humanact)}"
                )
            if full_run:
                result = verify_pose_tree(
                    contract,
                    args.staging_root / "pose_data",
                    state,
                    pose_fingerprint,
                    index_sha256,
                    device_record,
                    verify_hash=args.verify_resume_hash,
                )
                marker = args.staging_root / "manifests/pose-success.json"
                json_dump_atomic(marker, result)
                print(f"POSE_READY {marker}")


def run_verify_pose(args: argparse.Namespace) -> None:
    validate_layout(
        args.source_root, args.archive_root, args.staging_root, args.data_root
    )
    configure_runtime_environment(args.staging_root)
    force_cpu_only_environment()
    preflight = load_preflight(args.preflight_manifest, args)
    pose_fingerprint = validate_sha256(
        preflight["pose_fingerprint"], label="pose fingerprint"
    )
    index_sha256 = validate_sha256(
        preflight["index"]["sha256"], label="index.csv"
    )
    contract = load_index_contract(args.source_root / "index.csv")
    with FileLock(args.staging_root / ".pipeline.lock"):
        with StateDB(args.staging_root / "state.sqlite3") as state:
            result = verify_pose_tree(
                contract,
                args.staging_root / "pose_data",
                state,
                pose_fingerprint,
                index_sha256,
                {"type": "cpu"},
                verify_hash=True,
            )
            json_dump_atomic(
                args.staging_root / "manifests/pose-success.json", result
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def run_status(args: argparse.Namespace) -> None:
    validate_layout(
        args.source_root, args.archive_root, args.staging_root, args.data_root
    )
    configure_runtime_environment(args.staging_root)
    force_cpu_only_environment()
    state_path = args.staging_root / "state.sqlite3"
    counts: dict[str, int] = {}
    if state_path.is_file():
        with StateDB(state_path) as state:
            counts = state.counts()
    print(
        json.dumps(
            {
                "staging_root": str(args.staging_root),
                "preflight_ready": (
                    args.staging_root / "manifests/preflight.json"
                ).is_file(),
                "pose_ready": (
                    args.staging_root / "manifests/pose-success.json"
                ).is_file(),
                "state_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=STAGING_ROOT)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    add_paths(preflight)
    preflight.add_argument(
        "--archive-manifest", type=Path, default=ARCHIVE_MANIFEST
    )
    preflight.add_argument("--reserve-gib", type=float, default=20.0)
    preflight.add_argument(
        "--projected-output-gib", type=float, default=12.0
    )
    preflight.add_argument("--skip-model-init", action="store_true")
    preflight.add_argument(
        "--allow-concurrent-archive-readers", action="store_true"
    )
    preflight.set_defaults(func=run_preflight)
    pose = sub.add_parser("pose")
    add_paths(pose)
    pose.add_argument("--preflight-manifest", type=Path)
    pose.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    pose.add_argument("--gpu-id", type=int)
    pose.add_argument("--gpu-uuid")
    pose.add_argument("--batch-frames", type=int, default=256)
    pose.add_argument("--max-initial-gpu-memory-mib", type=int, default=256)
    pose.add_argument("--max-initial-gpu-util", type=int, default=5)
    pose.add_argument(
        "--verify-resume-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    pose.add_argument(
        "--only-archive",
        action="append",
        choices=tuple(ARCHIVE_TOPS) + ("HumanAct12",),
    )
    pose.set_defaults(func=run_pose)
    verify = sub.add_parser("verify-pose")
    add_paths(verify)
    verify.add_argument("--preflight-manifest", type=Path)
    verify.set_defaults(func=run_verify_pose)
    status_parser = sub.add_parser("status")
    add_paths(status_parser)
    status_parser.set_defaults(func=run_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command in ("pose", "verify-pose") and args.preflight_manifest is None:
        args.preflight_manifest = (
            args.staging_root / "manifests/preflight.json"
        )
    try:
        args.func(args)
    except (PipelineError, FileNotFoundError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
