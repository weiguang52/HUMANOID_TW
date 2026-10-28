"""Paired contact instrumentation; candidate replaces only the slip reward."""
from isaaclab.managers import RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp.point_slip import ground_point_slip
from .stability_env_cfg import HeadLockedEnvCfg

@configclass
class SlipBaselineEnvCfg(HeadLockedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for foot in ('left_foot','right_foot'):
            setattr(self.scene,'slip_'+foot,ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Robot/'+foot,update_period=0.,
                filter_prim_paths_expr=['/World/ground/terrain/GroundPlane/CollisionPlane'],
                track_contact_points=True,max_contact_data_count_per_prim=64))

@configclass
class PointSlipEnvCfg(SlipBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_slide=RewardTermCfg(func=ground_point_slip,weight=self.rewards.feet_slide.weight)

class SlipBaselinePlayEnvCfg(SlipBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9

class PointSlipPlayEnvCfg(PointSlipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9

@configclass
class MovingSlipEnvCfg(SlipBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from unitree_rl_lab.tasks.mimic.mdp.point_slip import normalized_moving_point_slip
        self.rewards.moving_point_slip=RewardTermCfg(func=normalized_moving_point_slip,weight=-.2,
            params={'command_name':'motion','speed_scale':.05})
        from unitree_rl_lab.tasks.mimic.mdp.leg_velocity_error import moving_leg_velocity_error
        self.rewards.moving_leg_velocity=RewardTermCfg(func=moving_leg_velocity_error,weight=0.,
            params={'command_name':'motion','scale':1.0})
        from unitree_rl_lab.tasks.mimic.mdp.leg_position_error import moving_leg_position_error
        self.rewards.moving_leg_position=RewardTermCfg(func=moving_leg_position_error,weight=0.,
            params={'command_name':'motion','scale':0.5})
        from unitree_rl_lab.tasks.mimic.mdp.leg_error_highpass import moving_leg_highpass_error
        self.rewards.moving_leg_highpass=RewardTermCfg(func=moving_leg_highpass_error,weight=0.,
            params={'command_name':'motion','cutoff_hz':8.0,'scale':.01,'filter_order':1,'trace_steps':0})

class MovingSlipPlayEnvCfg(MovingSlipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs=1
        self.episode_length_s=1.e9
