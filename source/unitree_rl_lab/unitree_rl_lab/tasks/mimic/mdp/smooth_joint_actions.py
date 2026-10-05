"""Optional physical-unit target shaping, shared by training and replay."""
import math
import torch
from .target_rate_scales import resolve_scales
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass


class SmoothJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.smoothing_tau) or cfg.smoothing_tau < 0:
            raise ValueError('smoothing_tau must be finite and nonnegative')
        if not math.isfinite(cfg.target_velocity_scale) or not 0 < cfg.target_velocity_scale <= 1:
            raise ValueError("target_velocity_scale must be in (0,1]")
        names = (self._asset.joint_names[self._joint_ids] if isinstance(self._joint_ids, slice)
                 else [self._asset.joint_names[i] for i in self._joint_ids])
        self._target_scales = torch.tensor(resolve_scales(names, cfg.target_velocity_scale,
            getattr(cfg, 'target_velocity_scale_by_joint', None)), device=self.device)
        self._target_dt = env.step_dt
        self._alpha = 1.0 if cfg.smoothing_tau == 0 else -math.expm1(-env.step_dt / cfg.smoothing_tau)
        self._target_memory = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        bounded = getattr(self.cfg, 'limit_target_position', False)
        if bounded:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            lower, upper = limits[..., 0], limits[..., 1]
            self._processed_actions = self._processed_actions.clamp(min=lower, max=upper)
        if self.cfg.smoothing_tau == 0 and not self.cfg.limit_target_velocity:
            return
        missing = ~self._target_initialized
        self._target_memory[missing] = self._asset.data.joint_pos[missing][:, self._joint_ids]
        if bounded:
            # Also bound state after reset; never accumulate inaccessible targets.
            self._target_memory.clamp_(min=lower, max=upper)
        self._target_initialized[:] = True
        change = self._alpha * (self._processed_actions - self._target_memory)
        if self.cfg.limit_target_velocity:
            maximum = self._asset.data.joint_vel_limits[:, self._joint_ids] * self._target_dt * self._target_scales
            change = torch.clamp(change, min=-maximum, max=maximum)
        self._target_memory.add_(change)
        if bounded:
            self._target_memory.clamp_(min=lower, max=upper)
        self._processed_actions = self._target_memory.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # Seed after all reset events have placed the robot, on the next command.
        self._target_initialized[env_ids] = False


@configclass
class SmoothJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SmoothJointPositionAction
    smoothing_tau: float = 0.0
    limit_target_velocity: bool = False
    target_velocity_scale: float = 1.0
    target_velocity_scale_by_joint: dict[str, float] | None = None
    limit_target_position: bool = False
