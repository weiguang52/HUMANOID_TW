from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.sole_tracking import sole_tracking_cost
from .path_env_cfg import PathRateEnvCfg
from .size_env_cfg import LEG_LENGTH_M

@configclass
class SoleEnvCfg(PathRateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.sole_height_residual=RewTerm(func=sole_tracking_cost,weight=-.5,
            params={'command_name':'motion','scale':.05*LEG_LENGTH_M,
            'urdf_path':'/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf'})

class SolePlayEnvCfg(SoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9


@configclass
class SoleMaskedEnvCfg(SoleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.sole_height_residual.params["exclude_known_stance"]=True

class SoleMaskedPlayEnvCfg(SoleMaskedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
