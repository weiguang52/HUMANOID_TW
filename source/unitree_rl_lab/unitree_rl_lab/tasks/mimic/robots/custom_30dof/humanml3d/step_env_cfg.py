"""TW140 paired experiment: identical step goals, only velocity width differs."""
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic import mdp
from unitree_rl_lab.tasks.mimic.mdp import step_tracking
from .balanced_env_cfg import BoundedExpressionEnvCfg

LEGS = [f'{side}_{joint}_joint' for side in ('left','right') for joint in
        ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_yaw','ankle_pitch')]
FEET = ['left_foot', 'right_foot']


@configclass
class StepEnvCfg(BoundedExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for group in (self.observations.policy, self.observations.critic):
            group.leg_preview = ObsTerm(func=step_tracking.leg_preview,
                params={'command_name':'motion','joint_names':LEGS})
        self.rewards.leg_joint_tracking = RewTerm(
            func=mdp.motion_joint_position_error_exp, weight=.75,
            params={'command_name':'motion','joint_names':LEGS,'std':.35})
        self.rewards.motion_foot_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp, weight=1.5,
            params={'command_name':'motion','body_names':FEET,'std':.04})
        self.rewards.swing_height = RewTerm(
            func=step_tracking.swing_height_tracking, weight=.5,
            params={'command_name':'motion','body_names':FEET,'std':.02})


@configclass
class StepVelocityEnvCfg(StepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_body_lin_vel.params['std'] = .10


class StepPlayEnvCfg(StepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9


class StepVelocityPlayEnvCfg(StepVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
