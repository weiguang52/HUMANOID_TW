from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.standing_heading import stationary_heading
from .stability_env_cfg import HeadLockedEnvCfg

@configclass
class StandingHeadingEnvCfg(HeadLockedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stationary_heading=RewTerm(func=stationary_heading,weight=-0.05,
            params={'command_name':'motion','scale':0.35})

class StandingHeadingPlayEnvCfg(StandingHeadingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
