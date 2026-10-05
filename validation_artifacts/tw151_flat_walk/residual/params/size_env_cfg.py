"""TW141: scales anchored to the audited small-robot FK geometry.

Lengths are joint-origin chain lengths, not mesh extents or user height.
Audit provenance: validation_artifacts/tw141_setup/scale_audit.json.
No automatic morphology changes, contact warping or angle rescaling.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.size_tracking import foot_residual_cost
from .step_env_cfg import StepEnvCfg, FEET

LEG_LENGTH_M = 0.2085258588194847
ARM_LENGTH_M = 0.09895936772227287
FOOT_SCALE_M = .08 * LEG_LENGTH_M


@configclass
class SizeEnvCfg(StepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_foot_pos.params['std'] = FOOT_SCALE_M
        self.rewards.swing_height.params['std'] = .05 * LEG_LENGTH_M
        self.rewards.motion_global_anchor_pos.params['std'] = .10 * LEG_LENGTH_M
        self.rewards.motion_core_pos.params['std'] = .20 * LEG_LENGTH_M
        self.rewards.motion_arm_pos.params['std'] = .20 * ARM_LENGTH_M
        self.rewards.standing_drift.params['std'] = .10 * LEG_LENGTH_M
        # Leg lengths per second; audit walking median speed is 0.0534 m/s.
        self.rewards.motion_body_lin_vel.params['std'] = .25 * LEG_LENGTH_M


@configclass
class SizeResidualEnvCfg(SizeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_residual = RewTerm(func=foot_residual_cost, weight=-.5,
            params={'command_name':'motion','body_names':FEET,'scale':FOOT_SCALE_M})


class SizePlayEnvCfg(SizeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9


class SizeResidualPlayEnvCfg(SizeResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
