"""TW170: one added residual, otherwise exact PathRate continuation."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.swing_knee import swing_knee_cost
from .path_env_cfg import PathRateEnvCfg


@configclass
class KneeEnvCfg(PathRateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.swing_knee_residual = RewTerm(func=swing_knee_cost,
            weight=-.5, params={'command_name': 'motion', 'scale': .35})


class KneePlayEnvCfg(KneeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
