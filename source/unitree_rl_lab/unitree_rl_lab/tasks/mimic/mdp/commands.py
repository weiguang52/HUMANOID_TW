from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .adaptive_sampling import balanced_sampling_probabilities, failure_rates
from .motion_library import MotionLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        root_body_name = self.cfg.root_body_name or self.cfg.body_names[0]
        self.motion_root_body_index = self.cfg.body_names.index(root_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        joint_expressions = self.cfg.joint_names if self.cfg.joint_names is not None else [".*"]
        joint_indexes, resolved_joint_names = self.robot.find_joints(joint_expressions, preserve_order=True)
        self.joint_indexes = torch.tensor(joint_indexes, dtype=torch.long, device=self.device)
        if self.cfg.joint_names is not None and resolved_joint_names != self.cfg.joint_names:
            raise ValueError(
                "Configured motion joint order did not resolve exactly. "
                f"expected={self.cfg.joint_names}, resolved={resolved_joint_names}"
            )

        control_fps = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        self.motion = MotionLibrary(
            self.cfg.motion_file,
            device=self.device,
            tracked_body_names=self.cfg.body_names,
            joint_names=self.cfg.joint_names,
            legacy_body_indexes=self.body_indexes,
            expected_fps=control_fps,
            max_frames=self.cfg.max_motion_frames,
            require_quality_pass=self.cfg.require_quality_pass,
        )
        if self.motion.joint_pos.shape[1] != len(self.joint_indexes):
            raise ValueError(
                f"Motion has {self.motion.joint_pos.shape[1]} joints, but robot selection "
                f"resolved {len(self.joint_indexes)} joints."
            )
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._evaluation_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = self.motion.build_bins(self.cfg.adaptive_bin_size_s)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.bin_visit_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_visited = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._sample_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_sampling_probability = self.cfg.adaptive_max_probability
        if self.max_sampling_probability is not None:
            self.max_sampling_probability = max(self.max_sampling_probability, 1.0 / self.bin_count)
        if self.motion.num_motions > 1 and self.cfg.adaptive_kernel_size != 1:
            raise ValueError("Multi-motion adaptive sampling currently requires adaptive_kernel_size=1")
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        self.metrics['sampling_effective_bins'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['failure_rate_mean'] = torch.zeros(self.num_envs, device=self.device)
        self.metrics['failure_rate_max'] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def frame_indices(self) -> torch.Tensor:
        return self.motion.frame_indices(self.motion_ids, self.time_steps)

    def set_evaluation_motion_ids(self, motion_ids: Sequence[int] | torch.Tensor) -> None:
        """Pin every environment to one motion and restart it from frame zero on reset.

        This is intentionally an explicit runtime override instead of a training
        configuration option. It keeps the adaptive sampler unchanged during
        training while allowing deterministic per-clip evaluation.
        """
        values = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if values.ndim != 1 or values.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected one evaluation motion id per environment ({self.num_envs}), got {tuple(values.shape)}"
            )
        if torch.any(values < 0) or torch.any(values >= self.motion.num_motions):
            minimum = int(values.min().item()) if values.numel() else -1
            maximum = int(values.max().item()) if values.numel() else -1
            raise ValueError(
                f"Evaluation motion ids must be in [0, {self.motion.num_motions}), got [{minimum}, {maximum}]"
            )
        self._evaluation_motion_ids.copy_(values)
        self._sample_active.zero_()

    def clear_evaluation_motion_ids(self) -> None:
        """Return all environments to adaptive motion sampling on their next reset."""
        self._evaluation_motion_ids.fill_(-1)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.frame_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.frame_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_indices] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_indices, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_indices, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        previous_active = self._sample_active[env_ids]
        current_bin_index = self.motion.current_bin_indexes(self.motion_ids, self.time_steps)
        if torch.any(previous_active):
            completed_bins = current_bin_index[env_ids][previous_active]
            self._current_bin_visited[:] = torch.bincount(completed_bins, minlength=self.bin_count)
            fail_bins = current_bin_index[env_ids][previous_active & episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        failure_score = failure_rates(
            self.bin_failed_count,
            self.bin_visit_count,
            unvisited_score=self.cfg.adaptive_unvisited_score,
        )
        if self.cfg.adaptive_kernel_size > 1:
            failure_score = torch.nn.functional.pad(
                failure_score.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            failure_score = torch.nn.functional.conv1d(
                failure_score, self.kernel.view(1, 1, -1)
            ).view(-1)

        sampling_probabilities = balanced_sampling_probabilities(
            failure_score,
            self.motion.bin_weights,
            uniform_ratio=self.cfg.adaptive_uniform_ratio,
            max_probability=self.max_sampling_probability,
        )
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        fractions = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        sampled_motion_ids, sampled_time_steps = self.motion.sample_from_bins(sampled_bins, fractions)
        self.motion_ids[env_ids] = sampled_motion_ids
        self.time_steps[env_ids] = sampled_time_steps
        self._sample_active[env_ids] = True

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.zeros_like(H)
        pmax, imax = sampling_probabilities.max(dim=0)
        effective_bins = 1.0 / sampling_probabilities.square().sum()
        self.metrics['sampling_effective_bins'][:] = effective_bins / self.bin_count
        self.metrics['failure_rate_mean'][:] = failure_score.mean()
        self.metrics['failure_rate_max'][:] = failure_score.max()
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        fixed_mask = self._evaluation_motion_ids[env_ids_tensor] >= 0
        sampled_env_ids = env_ids_tensor[~fixed_mask]
        fixed_env_ids = env_ids_tensor[fixed_mask]

        if sampled_env_ids.numel() > 0:
            self._adaptive_sampling(sampled_env_ids)
        if fixed_env_ids.numel() > 0:
            self.motion_ids[fixed_env_ids] = self._evaluation_motion_ids[fixed_env_ids]
            self.time_steps[fixed_env_ids] = 0
            self._sample_active[fixed_env_ids] = False

        root_pos = self.body_pos_w[:, self.motion_root_body_index].clone()
        root_ori = self.body_quat_w[:, self.motion_root_body_index].clone()
        root_lin_vel = self.body_lin_vel_w[:, self.motion_root_body_index].clone()
        root_ang_vel = self.body_ang_vel_w[:, self.motion_root_body_index].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_indexes]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids],
            joint_vel[env_ids],
            joint_ids=self.joint_indexes,
            env_ids=env_ids,
        )
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # Public env.reset() does not call command.compute() before the first
        # transition. Seed these buffers with the newly selected reference.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_command(self):
        self.time_steps += 1
        clip_lengths = self.motion.clip_lengths[self.motion_ids]
        env_ids = torch.where(self.time_steps >= clip_lengths)[0]
        if self.cfg.resample_at_motion_end:
            self._resample_command(env_ids)
        else:
            self.time_steps = torch.minimum(self.time_steps, clip_lengths - 1)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self.bin_visit_count = (
            self.cfg.adaptive_alpha * self._current_bin_visited + (1 - self.cfg.adaptive_alpha) * self.bin_visit_count
        )
        self._current_bin_failed.zero_()
        self._current_bin_visited.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    root_body_name: str | None = None
    joint_names: list[str] | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    max_motion_frames: int = 2_000_000
    resample_at_motion_end: bool = True
    require_quality_pass: bool = False

    adaptive_bin_size_s: float = 1.0
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_unvisited_score: float = 1.0
    adaptive_max_probability: float | None = None

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
