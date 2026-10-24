"""Opt-in reward ablation; keep HeadLocked observations/actions/dynamics."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.support_velocity import stationary_support_velocity
from .stability_env_cfg import HeadLockedEnvCfg


@configclass
class SupportVelocityEnvCfg(HeadLockedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stationary_support_velocity = RewTerm(
            func=stationary_support_velocity, weight=-0.25,
            params={'command_name': 'motion', 'scale': 0.02},
        )


class SupportVelocityPlayEnvCfg(SupportVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
