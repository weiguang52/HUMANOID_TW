"""TW-specific reward terms (kept out of the shared locomotion mdp on purpose)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_swing_height(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    threshold: float,
    target_height: float,
    rest_height: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward lifting each foot during its swing half-cycle.

    Uses the same gait clock as mdp.feet_gait: a foot is in stance while its phase < threshold and in swing
    otherwise. During swing the reward grows linearly with the foot's height above its resting height and
    saturates at target_height. Nothing is paid in stance, so a stationary robot gets no reward from this
    term (unlike mdp.foot_clearance_reward, which is maximal when the feet do not move).
    Gated on |command| > 0.1 so it only applies while the robot is asked to move.
    """
    asset = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = ((foot_z - rest_height) / target_height).clamp(0.0, 1.0)

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    offsets = torch.tensor(offset, device=env.device, dtype=global_phase.dtype).unsqueeze(0)
    leg_phase = (global_phase + offsets) % 1.0
    is_swing = leg_phase >= threshold

    reward = torch.sum(height * is_swing, dim=1)
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward = reward * (cmd_norm > 0.1)
    return reward