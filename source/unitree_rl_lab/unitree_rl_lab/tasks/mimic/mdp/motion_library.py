"""Validated single- and multi-clip motion loading for mimic tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


MOTION_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _string_list(value: np.ndarray | None) -> list[str] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"Name metadata must be one-dimensional, got {array.shape}")
    return [str(item) for item in array.tolist()]


class MotionLibrary:
    """Eager motion library with explicit clip boundaries.

    This backend intentionally has a frame cap. It is suitable for the first
    filtered HumanML3D curriculum and keeps the command interface ready for a
    future sharded/cache backend without changing the RL task.
    """

    def __init__(
        self,
        source: str,
        *,
        device: str,
        tracked_body_names: Sequence[str],
        joint_names: Sequence[str] | None,
        legacy_body_indexes: Sequence[int] | torch.Tensor,
        expected_fps: float,
        max_frames: int,
        require_quality_pass: bool = False,
    ):
        self.source = Path(source).expanduser().resolve()
        self.device = device
        self.tracked_body_names = list(tracked_body_names)
        self.joint_names = None if joint_names is None else list(joint_names)
        if isinstance(legacy_body_indexes, torch.Tensor):
            legacy_body_indexes = legacy_body_indexes.detach().cpu().tolist()
        self.legacy_body_indexes = list(legacy_body_indexes)
        self.expected_fps = float(expected_fps)
        self.max_frames = int(max_frames)
        self.require_quality_pass = bool(require_quality_pass)
        if not np.isfinite(self.expected_fps) or self.expected_fps <= 0.0:
            raise ValueError(f"expected_fps must be finite and positive, got {expected_fps}")
        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")

        entries = self._read_entries()
        arrays: dict[str, list[torch.Tensor]] = {field: [] for field in MOTION_FIELDS}
        clip_ids: list[str] = []
        clip_lengths: list[int] = []
        clip_weights: list[float] = []
        fps_value: float | None = None
        total_frames = 0

        for entry in entries:
            path = entry["path"]
            weight = float(entry["weight"])
            motion_id = str(entry["id"])
            loaded = self._load_npz(path)
            clip_fps = loaded.pop("fps")
            if fps_value is None:
                fps_value = clip_fps
            elif abs(clip_fps - fps_value) > 1.0e-6:
                raise ValueError(f"Mixed motion FPS: {fps_value} and {clip_fps} in {path}")

            length = int(loaded["joint_pos"].shape[0])
            total_frames += length
            if total_frames > self.max_frames:
                raise MemoryError(
                    f"Motion manifest has {total_frames} frames, over eager limit "
                    f"{self.max_frames}. Filter the curriculum or use a sharded backend."
                )

            for field in MOTION_FIELDS:
                arrays[field].append(torch.from_numpy(loaded[field]))
            clip_ids.append(motion_id)
            clip_lengths.append(length)
            clip_weights.append(weight)

        if fps_value is None:
            raise ValueError(f"No motions found in {self.source}")
        if abs(fps_value - self.expected_fps) > 1.0e-4:
            raise ValueError(
                f"Motion FPS {fps_value:g} does not match environment control FPS "
                f"{self.expected_fps:g}. Convert motions before training."
            )

        self.fps = fps_value
        self.clip_ids = clip_ids
        self.num_motions = len(clip_ids)
        self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=device)
        self.clip_weights = torch.tensor(clip_weights, dtype=torch.float32, device=device)
        starts = np.cumsum([0, *clip_lengths[:-1]], dtype=np.int64)
        self.clip_starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.time_step_total = total_frames

        for field in MOTION_FIELDS:
            value = torch.cat(arrays[field], dim=0).to(device=device, dtype=torch.float32)
            setattr(self, field, value)

        self.bin_motion_ids: torch.Tensor | None = None
        self.bin_local_starts: torch.Tensor | None = None
        self.bin_widths: torch.Tensor | None = None
        self.bin_weights: torch.Tensor | None = None
        self.clip_bin_offsets: torch.Tensor | None = None
        self.frames_per_bin: int | None = None

    def _read_entries(self) -> list[dict[str, object]]:
        if self.source.suffix.lower() != ".json":
            if not self.source.is_file():
                raise FileNotFoundError(self.source)
            return [{"id": self.source.stem, "path": self.source, "weight": 1.0}]

        payload = json.loads(self.source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported motion manifest schema in {self.source}")
        motions = payload.get("motions")
        if not isinstance(motions, list) or not motions:
            raise ValueError(f"Manifest has no motions: {self.source}")

        entries: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(motions):
            if not isinstance(item, dict) or "file" not in item:
                raise ValueError(f"Invalid motion entry {index} in {self.source}")
            path = Path(str(item["file"])).expanduser()
            if not path.is_absolute():
                path = (self.source.parent / path).resolve()
            motion_id = str(item.get("id", path.stem))
            if motion_id in seen_ids:
                raise ValueError(f"Duplicate motion id: {motion_id}")
            seen_ids.add(motion_id)
            if self.require_quality_pass and item.get("quality_pass") is not True:
                raise ValueError(
                    f"Motion {motion_id} is not marked quality_pass=true; refusing training data."
                )
            weight = float(item.get("weight", 1.0))
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"Invalid sampling weight for {motion_id}: {weight}")
            entries.append({"id": motion_id, "path": path, "weight": weight})
        return entries

    def _load_npz(self, path: Path) -> dict[str, np.ndarray | float]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            missing = [field for field in MOTION_FIELDS if field not in data]
            if missing or "fps" not in data:
                raise ValueError(f"{path} is missing fields: {missing + ([] if 'fps' in data else ['fps'])}")

            fps_array = np.asarray(data["fps"]).reshape(-1)
            if fps_array.size != 1:
                raise ValueError(f"{path}: fps must be scalar, got {fps_array.shape}")
            fps = float(fps_array[0])
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"{path}: fps must be finite and positive, got {fps}")

            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
            body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
            stored_joint_names = _string_list(data["joint_names"] if "joint_names" in data else None)
            stored_body_names = _string_list(data["body_names"] if "body_names" in data else None)

        if joint_pos.ndim != 2 or joint_pos.shape[0] < 2:
            raise ValueError(f"{path}: joint_pos must be [T,J] with T>=2, got {joint_pos.shape}")
        length = joint_pos.shape[0]
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path}: joint velocity shape {joint_vel.shape} != {joint_pos.shape}")
        for name, array, tail in (
            ("body_pos_w", body_pos, 3),
            ("body_quat_w", body_quat, 4),
            ("body_lin_vel_w", body_lin_vel, 3),
            ("body_ang_vel_w", body_ang_vel, 3),
        ):
            if array.ndim != 3 or array.shape[0] != length or array.shape[2] != tail:
                raise ValueError(f"{path}: invalid {name} shape {array.shape}")
        body_count = body_pos.shape[1]
        for name, array in (
            ("body_quat_w", body_quat),
            ("body_lin_vel_w", body_lin_vel),
            ("body_ang_vel_w", body_ang_vel),
        ):
            if array.shape[1] != body_count:
                raise ValueError(
                    f"{path}: {name} body count {array.shape[1]} != body_pos_w count {body_count}"
                )
        if stored_joint_names is not None and len(stored_joint_names) != joint_pos.shape[1]:
            raise ValueError(
                f"{path}: joint_names count {len(stored_joint_names)} != joint array width {joint_pos.shape[1]}"
            )
        if stored_body_names is not None and len(stored_body_names) != body_count:
            raise ValueError(
                f"{path}: body_names count {len(stored_body_names)} != body array width {body_count}"
            )

        if self.joint_names is not None:
            if stored_joint_names is None:
                raise ValueError(f"{path}: custom motions require joint_names metadata")
            joint_indexes = self._ordered_indexes(stored_joint_names, self.joint_names, "joint", path)
            joint_pos = joint_pos[:, joint_indexes]
            joint_vel = joint_vel[:, joint_indexes]

        if stored_body_names is None:
            body_indexes = self.legacy_body_indexes
            if body_indexes and (min(body_indexes) < 0 or max(body_indexes) >= body_count):
                raise ValueError(
                    f"{path}: legacy body indexes {body_indexes} exceed body count {body_count}"
                )
        else:
            body_indexes = self._ordered_indexes(stored_body_names, self.tracked_body_names, "body", path)
        body_pos = body_pos[:, body_indexes]
        body_quat = body_quat[:, body_indexes]
        body_lin_vel = body_lin_vel[:, body_indexes]
        body_ang_vel = body_ang_vel[:, body_indexes]

        values = (joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError(f"{path}: motion contains NaN or Inf")
        quat_norm = np.linalg.norm(body_quat, axis=-1)
        max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
        if max_quat_error > 5.0e-3:
            raise ValueError(f"{path}: body quaternion norm error {max_quat_error:.6g}")

        return {
            "fps": fps,
            "joint_pos": np.ascontiguousarray(joint_pos),
            "joint_vel": np.ascontiguousarray(joint_vel),
            "body_pos_w": np.ascontiguousarray(body_pos),
            "body_quat_w": np.ascontiguousarray(body_quat),
            "body_lin_vel_w": np.ascontiguousarray(body_lin_vel),
            "body_ang_vel_w": np.ascontiguousarray(body_ang_vel),
        }

    @staticmethod
    def _ordered_indexes(
        stored: Sequence[str], requested: Sequence[str], kind: str, path: Path
    ) -> list[int]:
        duplicate = {name for name in stored if stored.count(name) > 1}
        if duplicate:
            raise ValueError(f"{path}: duplicate {kind} names: {sorted(duplicate)}")
        lookup = {name: index for index, name in enumerate(stored)}
        missing = [name for name in requested if name not in lookup]
        if missing:
            raise ValueError(f"{path}: missing requested {kind} names: {missing}")
        return [lookup[name] for name in requested]

    def frame_indices(self, motion_ids: torch.Tensor, local_frames: torch.Tensor) -> torch.Tensor:
        return self.clip_starts[motion_ids] + local_frames

    def build_bins(self, bin_size_s: float) -> int:
        if not np.isfinite(bin_size_s) or bin_size_s <= 0.0:
            raise ValueError(f"adaptive_bin_size_s must be positive, got {bin_size_s}")
        self.frames_per_bin = max(1, int(round(self.fps * bin_size_s)))
        motion_ids: list[int] = []
        local_starts: list[int] = []
        widths: list[int] = []
        weights: list[float] = []
        offsets: list[int] = []
        cursor = 0
        for motion_id, (length, weight) in enumerate(
            zip(self.clip_lengths.detach().cpu().tolist(), self.clip_weights.detach().cpu().tolist())
        ):
            offsets.append(cursor)
            # The last frame is a timeout target, never a normal episode start.
            startable_length = max(1, length - 1)
            for start in range(0, startable_length, self.frames_per_bin):
                motion_ids.append(motion_id)
                local_starts.append(start)
                width = min(self.frames_per_bin, startable_length - start)
                widths.append(width)
                # Weight by startable frames so partial tail bins are not oversampled.
                weights.append(weight * width)
                cursor += 1
        self.bin_motion_ids = torch.tensor(motion_ids, dtype=torch.long, device=self.device)
        self.bin_local_starts = torch.tensor(local_starts, dtype=torch.long, device=self.device)
        self.bin_widths = torch.tensor(widths, dtype=torch.long, device=self.device)
        self.bin_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.clip_bin_offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        return cursor

    def current_bin_indexes(self, motion_ids: torch.Tensor, local_frames: torch.Tensor) -> torch.Tensor:
        if self.frames_per_bin is None or self.clip_bin_offsets is None:
            raise RuntimeError("build_bins() must be called first")
        last_startable = self.clip_lengths[motion_ids] - 2
        startable_frames = torch.minimum(local_frames, last_startable)
        return self.clip_bin_offsets[motion_ids] + torch.div(
            startable_frames, self.frames_per_bin, rounding_mode="floor"
        )

    def sample_from_bins(self, bin_ids: torch.Tensor, fractions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bin_motion_ids is None or self.bin_local_starts is None or self.bin_widths is None:
            raise RuntimeError("build_bins() must be called first")
        motion_ids = self.bin_motion_ids[bin_ids]
        offsets = torch.floor(fractions * self.bin_widths[bin_ids].float()).long()
        offsets = torch.minimum(offsets, self.bin_widths[bin_ids] - 1)
        return motion_ids, self.bin_local_starts[bin_ids] + offsets
