"""TW154: flat mixed motions + horizontal progress, isolated leg rate ablation."""
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp import path_tracking
from .size_env_cfg import SizeResidualEnvCfg, LEG_LENGTH_M
from .step_env_cfg import LEGS


@configclass
class PathEnvCfg(SizeResidualEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for g in (self.observations.policy,self.observations.critic):
            g.horizontal_path_error=ObsTerm(func=path_tracking.path_error_observation,
                params={'command_name':'motion','scale':LEG_LENGTH_M})
        self.rewards.horizontal_path_error=RewTerm(func=path_tracking.path_error_cost,
            weight=-.5,params={'command_name':'motion','scale':.5*LEG_LENGTH_M})
        # Horizontal cost also constrains standstill; avoid a duplicate standing-only cost.
        self.rewards.standing_drift=None


@configclass
class PathRateEnvCfg(PathEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.JointPositionAction.target_velocity_scale_by_joint={n:.75 for n in LEGS}


class PathPlayEnvCfg(PathEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9


class PathRatePlayEnvCfg(PathRateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
