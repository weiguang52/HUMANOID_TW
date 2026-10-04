"""TW100: upper-body expression + locomotion, retaining human contact timing.

Separate opt-in task. Original task/rewards and source data remain unchanged.
Observation dimensions differ: train fresh; do not resume a full-body checkpoint.
"""
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.mdp import expression
from .tracking_env_cfg import RobotEnvCfg, RETARGETED_JOINT_NAMES, ROOT_BODY, ANCHOR_BODY, WRIST_BODIES

UPPER_JOINTS = [n for n in RETARGETED_JOINT_NAMES if not any(
    token in n for token in ('hip_', 'knee_', 'ankle_', 'foot_roll'))]
UPPER_BODIES = [ANCHOR_BODY, 'left_upper_arm', 'left_force_arm', 'left_wrist',
                'right_upper_arm', 'right_force_arm', 'right_wrist']


@configclass
class ExpressionEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Keep the motion clock and original contact sidecar unchanged.
        for group, command_field in ((self.observations.policy, 'motion_command'),
                                     (self.observations.critic, 'command')):
            setattr(group, command_field, ObsTerm(func=expression.upper_command,
                params={'command_name': 'motion', 'joint_names': UPPER_JOINTS}))
            group.contact_phase = ObsTerm(func=expression.contact_command,
                params={'command_name': 'motion'})
            group.motion_anchor_pos_b = ObsTerm(func=expression.anchor_height_error,
                params={'command_name': 'motion'})
        r = self.rewards
        r.motion_leg_pos = None
        r.motion_foot_pos = None
        r.motion_global_anchor_pos = RewTerm(func=expression.height_tracking,
            weight=1., params={'command_name': 'motion', 'std': .08})
        r.standing_drift = RewTerm(func=expression.standing_drift,
            weight=-.5, params={'command_name': 'motion', 'std': .08})
        r.motion_arm_pos.weight = 1.5
        r.motion_arm_pos.params['std'] = .10
        r.motion_body_ori.params['body_names'] = UPPER_BODIES
        # Root/chest velocity goals remain; leg segment velocities are unconstrained.
        r.motion_body_lin_vel.params['body_names'] = [ROOT_BODY, ANCHOR_BODY]
        r.motion_body_lin_vel.params['std'] = .25
        r.motion_body_lin_vel.weight = 2.
        r.motion_body_ang_vel.params['body_names'] = [ROOT_BODY, ANCHOR_BODY]
        r.motion_joint_pos.params['joint_names'] = UPPER_JOINTS
        r.motion_joint_pos.weight = 1.
        r.motion_joint_vel_cost = RewTerm(func=expression.upper_velocity_cost,
            weight=-.2, params={'command_name': 'motion', 'joint_names': UPPER_JOINTS})
        r.contact_phase.weight = -.5
        r.feet_slide.weight = -.2
        r.action_rate_l2.weight = -.2
        r.joint_acc.weight = -2e-6
        # Keep height/orientation fall checks; leg trajectory deviations are allowed.
        self.terminations.ee_body_pos.params['body_names'] = WRIST_BODIES
        self.commands.motion.adaptive_uniform_ratio = 1.
        self.commands.motion.adaptive_max_probability = None
        self.actions.JointPositionAction.target_velocity_scale = .5
        self.actions.JointPositionAction.limit_target_velocity = True
        self.actions.JointPositionAction.smoothing_tau = 0.


class ExpressionPlayEnvCfg(ExpressionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.e9
