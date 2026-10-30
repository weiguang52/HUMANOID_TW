"""Reference-motion storage for SuperMimic.

A :class:`MotionLibrary` loads a set of mjlab motion ``.npz`` files (the format
produced by :mod:`supermimic.scripts.preprocess`) and concatenates
them so that reference frames can be gathered for many environments at once.
Only the bodies that are actually tracked are kept in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionLibraryCfg:
    motion_files: tuple[str, ...]
    body_names: tuple[str, ...]
    device: str = "cuda:0"
    max_frames: int | None = None
    joint_names: tuple[str, ...] = ()
    required_coordinate_contract: str | None = None


class MotionLibrary:
    """Concatenated, GPU-resident reference-motion library."""

    def __init__(self, cfg: MotionLibraryCfg) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dt = 1.0 / 50.0
        self.clip_names: list[str] = []
        self.clip_lengths: list[int] = []
        self.fps: float = 50.0

        joint_pos, joint_vel = [], []
        body_pos, body_quat, body_lin_vel, body_ang_vel = [], [], [], []

        for path_str in cfg.motion_files:
            path = Path(path_str)
            with np.load(path, allow_pickle=False) as data:
                if "body_names" in data:
                    file_bodies = tuple(str(x) for x in data["body_names"].tolist())
                    indices = [file_bodies.index(name) for name in cfg.body_names]
                else:
                    indices = list(range(len(cfg.body_names)))
                joint_indices = list(range(data["joint_pos"].shape[1]))
                if cfg.joint_names:
                    if "joint_names" not in data:
                        raise ValueError(f"{path}: missing joint_names metadata")
                    file_joints = tuple(str(x) for x in data["joint_names"].tolist())
                    if len(set(file_joints)) != len(file_joints) or set(
                        file_joints
                    ) != set(cfg.joint_names):
                        raise ValueError(
                            f"{path}: joint name set mismatch or duplicates"
                        )
                    if len(file_joints) != data["joint_pos"].shape[1]:
                        raise ValueError(f"{path}: joint metadata width mismatch")
                    joint_indices = [
                        file_joints.index(name) for name in cfg.joint_names
                    ]
                if cfg.required_coordinate_contract is not None:
                    if (
                        "body_names" not in data
                        or "joint_coordinate_contract" not in data
                    ):
                        raise ValueError(f"{path}: missing TW body/coordinate metadata")
                    contract = str(
                        np.asarray(data["joint_coordinate_contract"]).reshape(-1)[0]
                    )
                    if contract != cfg.required_coordinate_contract:
                        raise ValueError(
                            f"{path}: stale joint coordinate contract {contract}"
                        )
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if (
                    not np.isfinite(fps)
                    or fps <= 0
                    or (self.clip_names and fps != self.fps)
                ):
                    raise ValueError(f"{path}: invalid or mixed reference FPS")
                if cfg.required_coordinate_contract is not None:
                    frame_count = data["joint_pos"].shape[0]
                    if frame_count < 2:
                        raise ValueError(f"{path}: motion needs at least two frames")
                    for key in [
                        "joint_pos",
                        "joint_vel",
                        "body_pos_w",
                        "body_quat_w",
                        "body_lin_vel_w",
                        "body_ang_vel_w",
                    ]:
                        values = data[key]
                        if (
                            values.shape[0] != frame_count
                            or not np.isfinite(values).all()
                        ):
                            raise ValueError(f"{path}: inconsistent or nonfinite {key}")
                    if data["joint_vel"].shape != data["joint_pos"].shape:
                        raise ValueError(f"{path}: joint velocity shape mismatch")
                    norms = np.linalg.norm(data["body_quat_w"], axis=-1)
                    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0):
                        raise ValueError(f"{path}: invalid body quaternions")
                n = int(data["joint_pos"].shape[0])
                if cfg.max_frames is not None:
                    n = min(n, cfg.max_frames)
                self.clip_names.append(path.stem)
                self.clip_lengths.append(n)
                self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                joint_pos.append(
                    torch.as_tensor(
                        data["joint_pos"][:n][:, joint_indices], dtype=torch.float32
                    )
                )
                joint_vel.append(
                    torch.as_tensor(
                        data["joint_vel"][:n][:, joint_indices], dtype=torch.float32
                    )
                )
                body_pos.append(
                    torch.as_tensor(
                        data["body_pos_w"][:n][:, indices], dtype=torch.float32
                    )
                )
                body_quat.append(
                    torch.as_tensor(
                        data["body_quat_w"][:n][:, indices], dtype=torch.float32
                    )
                )
                body_lin_vel.append(
                    torch.as_tensor(
                        data["body_lin_vel_w"][:n][:, indices], dtype=torch.float32
                    )
                )
                body_ang_vel.append(
                    torch.as_tensor(
                        data["body_ang_vel_w"][:n][:, indices], dtype=torch.float32
                    )
                )

        if not joint_pos:
            raise ValueError("MotionLibrary received no motion files.")
        self.dt = 1.0 / self.fps
        self.num_clips = len(self.clip_names)
        self.clip_lengths_t = torch.tensor(
            self.clip_lengths, dtype=torch.long, device=self.device
        )
        self.offsets = torch.zeros(self.num_clips + 1, dtype=torch.long)
        self.offsets[1:] = torch.cumsum(self.clip_lengths_t.cpu(), dim=0)
        self.offsets = self.offsets.to(self.device)
        self.num_frames = int(self.offsets[-1].item())

        def _cat(xs: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(xs, dim=0).to(self.device, non_blocking=False)

        self.joint_pos = _cat(joint_pos)
        self.joint_vel = _cat(joint_vel)
        self.body_pos = _cat(body_pos)
        self.body_quat = _cat(body_quat)
        self.body_lin_vel = _cat(body_lin_vel)
        self.body_ang_vel = _cat(body_ang_vel)

    @property
    def num_bodies(self) -> int:
        return self.body_pos.shape[1]

    def global_index(self, clip_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        times = torch.clamp(times, min=0)
        lengths = self.clip_lengths_t[clip_ids] - 1
        times = torch.minimum(times, lengths)
        return self.offsets[clip_ids] + times

    def gather(
        self,
        clip_ids: torch.Tensor,
        times: torch.Tensor,
        horizon: int,
        offset: int = 0,
    ) -> dict[str, torch.Tensor]:
        batch = clip_ids.shape[0]
        frames = (
            times[:, None]
            + offset
            + torch.arange(horizon, device=clip_ids.device)[None, :]
        )
        idx = self.global_index(clip_ids[:, None].expand(batch, horizon), frames)
        return {
            "joint_pos": self.joint_pos[idx],
            "joint_vel": self.joint_vel[idx],
            "body_pos": self.body_pos[idx],
            "body_quat": self.body_quat[idx],
            "body_lin_vel": self.body_lin_vel[idx],
            "body_ang_vel": self.body_ang_vel[idx],
        }

    def gather_index(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "joint_pos": self.joint_pos[index],
            "joint_vel": self.joint_vel[index],
            "body_pos": self.body_pos[index],
            "body_quat": self.body_quat[index],
            "body_lin_vel": self.body_lin_vel[index],
            "body_ang_vel": self.body_ang_vel[index],
        }

    def clip_and_bin(
        self, clip_ids: torch.Tensor, times: torch.Tensor, bin_size_s: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bin_size = max(1, round(bin_size_s / self.dt))
        return clip_ids, times // bin_size

    @property
    def max_bins(self) -> int:
        return int(np.ceil(max(self.clip_lengths) * self.dt / 1.0))
