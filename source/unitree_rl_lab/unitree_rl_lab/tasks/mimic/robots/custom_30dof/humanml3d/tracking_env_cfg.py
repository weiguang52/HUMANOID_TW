from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.custom_humanoid import (
    CUSTOM_HUMANOID_30DOF_CFG as ROBOT_CFG,
    CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
    CUSTOM_HUMANOID_ACTION_SCALE,
    CUSTOM_HUMANOID_TRACKING_BODY_NAMES,
)


DATA_ROOT = os.environ.get("GPUFREE_DATA_ROOT", "/root/gpufree-data")
MOTION_MANIFEST = os.environ.get(
    "PRACTICE9_CUSTOM_MOTION_MANIFEST",
    f"{DATA_ROOT}/datasets/practice9/humanml3d_custom30/manifest.json",
)
ROOT_BODY = "base_link"
ANCHOR_BODY = "chest"
FOOT_BODIES = ["left_foot", "right_foot"]
WRIST_BODIES = ["left_wrist", "right_wrist"]
RETARGETED_JOINT_NAMES = [
    name for name in CUSTOM_HUMANOID_30DOF_JOINT_NAMES if name not in {"left_foot_roll", "right_foot_roll"}
]
JOINT_ENTITY = SceneEntityCfg(
    "robot",
    joint_names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
    preserve_order=True,
)

VELOCITY_RANGE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.10, 0.10),
    "roll": (-0.25, 0.25),
    "pitch": (-0.25, 0.25),
    "yaw": (-0.40, 0.40),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=MOTION_MANIFEST,
        anchor_body_name=ANCHOR_BODY,
        root_body_name=ROOT_BODY,
        body_names=CUSTOM_HUMANOID_TRACKING_BODY_NAMES,
        joint_names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
        resampling_time_range=(1.0e9, 1.0e9),
        resample_at_motion_end=False,
        max_motion_frames=500_000,
        require_quality_pass=os.environ.get("P9_CUSTOM_ALLOW_UNSAFE_MOTIONS", "0") != "1",
        adaptive_bin_size_s=1.0,
        adaptive_kernel_size=1,
        adaptive_uniform_ratio=0.7,
        adaptive_alpha=0.001,
        adaptive_unvisited_score=1.0,
        adaptive_max_probability=0.02,
        debug_vis=False,
        pose_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.10, 0.10),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.05, 0.05),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
        preserve_order=True,
        scale=CUSTOM_HUMANOID_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.10, n_max=0.10))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.15, n_max=0.15))
        anchor_lin_vel_error_b = ObsTerm(
            func=mdp.motion_anchor_lin_vel_error_b,
            params={"command_name": "motion"},
        )
        anchor_ang_vel_error_b = ObsTerm(
            func=mdp.motion_anchor_ang_vel_error_b,
            params={"command_name": "motion"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": JOINT_ENTITY},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": JOINT_ENTITY},
            noise=Unoise(n_min=-0.30, n_max=0.30),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": JOINT_ENTITY})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": JOINT_ENTITY})
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 32,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": JOINT_ENTITY,
            "pos_distribution_params": (-0.005, 0.005),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ANCHOR_BODY),
            "com_range": {"x": (-0.002, 0.002), "y": (-0.002, 0.002), "z": (-0.002, 0.002)},
        },
    )


@configclass
class RewardsCfg:
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": JOINT_ENTITY})
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": JOINT_ENTITY})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": JOINT_ENTITY})
    foot_roll_default = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_foot_roll", "right_foot_roll"],
                preserve_order=True,
            )
        },
    )

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.15},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_core_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.15, "body_names": [ROOT_BODY, ANCHOR_BODY]},
    )
    motion_leg_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.12,
            "body_names": [
                "left_thigh", "left_mid_leg", "left_foot",
                "right_thigh", "right_mid_leg", "right_foot",
            ],
        },
    )
    motion_arm_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.6,
        params={
            "command_name": "motion",
            "std": 0.18,
            "body_names": [
                "left_upper_arm", "left_force_arm", "left_wrist",
                "right_upper_arm", "right_force_arm", "right_wrist",
            ],
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.5},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.75},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 2.0},
    )
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.35,
        params={"command_name": "motion", "std": 0.5, "joint_names": RETARGETED_JOINT_NAMES},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot$)(?!right_foot$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, time_out=True, params={"command_name": "motion"})
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.12},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.12,
            "body_names": FOOT_BODIES + WRIST_BODIES,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
