"""Incremental reward ablations; observations/actions/dynamics stay compatible."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp import stability_tracking as reward
from .path_env_cfg import PathRateEnvCfg
from .size_env_cfg import LEG_LENGTH_M


@configclass
class PrecisionEnvCfg(PathRateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.horizontal_path_error = RewTerm(func=reward.precise_path_cost, weight=-.5,
            params={'command_name': 'motion', 'walking_scale': .5*LEG_LENGTH_M,
                    'standing_scale': .1*LEG_LENGTH_M, 'speed_threshold': .025})


@configclass
class PrecisionNeckEnvCfg(PrecisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.neck_reference_position = RewTerm(func=reward.neck_tracking_cost, weight=-.25,
            params={'command_name': 'motion', 'scale': .1745329252})
        self.rewards.neck_reference_velocity = RewTerm(func=reward.neck_tracking_cost, weight=-.05,
            params={'command_name': 'motion', 'scale': 1., 'velocity': True})


@configclass
class PrecisionNeckRootEnvCfg(PrecisionNeckEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.pelvis_reference_heading = RewTerm(func=reward.pelvis_heading_cost, weight=-.25,
            params={'command_name': 'motion', 'scale': .2617993878})


class PrecisionPlayEnvCfg(PrecisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9


class PrecisionNeckPlayEnvCfg(PrecisionNeckEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9


class PrecisionNeckRootPlayEnvCfg(PrecisionNeckRootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9

@configclass
class NeckOnlyEnvCfg(PathRateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.neck_reference_position = RewTerm(func=reward.neck_tracking_cost, weight=-.25,
            params={'command_name': 'motion', 'scale': .1745329252})
        self.rewards.neck_reference_velocity = RewTerm(func=reward.neck_tracking_cost, weight=-.05,
            params={'command_name': 'motion', 'scale': 1., 'velocity': True})



class NeckOnlyPlayEnvCfg(NeckOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
