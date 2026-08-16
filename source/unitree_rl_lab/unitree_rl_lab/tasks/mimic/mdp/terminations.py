from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
from unitree_rl_lab.tasks.mimic.mdp.rewards import _get_body_indexes


def _record_evaluation_failure(
    command: MotionCommand,
    term_name: str,
    failed: torch.Tensor,
    measure: torch.Tensor,
    body_names: list[str] | None = None,
) -> None:
    '''Send terminal-state measurements to an optional evaluation-only recorder.'''
    recorder = getattr(command, '_evaluation_failure_recorder', None)
    if recorder is not None:
        recorder.record(term_name, failed, measure, body_names)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    failed = error > threshold
    _record_evaluation_failure(command, 'anchor_pos', failed, error)
    return failed


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()
    failed = error > threshold
    _record_evaluation_failure(command, 'anchor_ori', failed, error)
    return failed


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    failed = torch.any(error > threshold, dim=-1)
    selected_names = [command.cfg.body_names[index] for index in body_indexes]
    _record_evaluation_failure(command, 'ee_body_pos', failed, error, selected_names)
    return failed


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End an episode cleanly at the last local frame of each motion clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    lengths = command.motion.clip_lengths[command.motion_ids]
    return command.time_steps >= lengths - 1
