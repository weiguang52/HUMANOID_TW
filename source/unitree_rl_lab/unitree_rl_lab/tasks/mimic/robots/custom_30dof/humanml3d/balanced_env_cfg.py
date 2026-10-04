"""TW108: joint-limit-safe expression policies, mixed walking/standing data."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.joint_compensation import group_target_error
from .expression_env_cfg import ExpressionEnvCfg

GROUPS = {
    'feet': (['left_foot_roll', 'right_foot_roll'], .05),
    'trunk': (['waist_yaw', 'gearbox_roll', 'chest_pitch'], .05),
    'arms': ([f'{side}_{joint}' for side in ('left','right') for joint in
        ('shoulder_pitch_joint','shoulder_roll_joint','shoulder_yaw_joint',
         'elbow_pitch_joint','wrist_pitch')], .02),
    'legs': ([f'{side}_{joint}_joint' for side in ('left','right') for joint in
        ('hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_yaw','ankle_pitch')], .02),
    'neck': (['neck','neck_linkage_roll','head_pitch'], .01),
}


@configclass
class BoundedExpressionEnvCfg(ExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.JointPositionAction.limit_target_position = True


@configclass
class BalancedExpressionEnvCfg(BoundedExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, (joints, weight) in GROUPS.items():
            setattr(self.rewards, 'compensation_' + name, RewTerm(
                func=group_target_error, weight=-weight,
                params={'joint_names': joints, 'scale': .25}))


class BoundedExpressionPlayEnvCfg(BoundedExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9


class BalancedExpressionPlayEnvCfg(BalancedExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
