"""Single reward ablation; reference timing, policy observations and dynamics unchanged."""
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.standing_contact import stationary_contact_loss
from .stability_env_cfg import HeadLockedEnvCfg

@configclass
class StandingContactEnvCfg(HeadLockedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stationary_contact_loss=RewTerm(func=stationary_contact_loss,weight=-.25,
            params={'command_name':'motion','sensor_cfg':SceneEntityCfg('contact_forces',
            body_names=['left_foot','right_foot'],preserve_order=True)})

class StandingContactPlayEnvCfg(StandingContactEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
