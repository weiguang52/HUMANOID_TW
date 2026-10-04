"""Isaac Lab asset configuration for the urdf0711 custom humanoid."""

from __future__ import annotations

import os

from unitree_rl_lab.assets.robots.custom_joint_coordinates import COORDINATES

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg


DATA_ROOT = os.environ.get("GPUFREE_DATA_ROOT", "/root/gpufree-data")
CUSTOM_HUMANOID_URDF = os.environ.get(
    "PRACTICE9_CUSTOM_URDF",
    f"{DATA_ROOT}/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf",
)
CUSTOM_HUMANOID_USD_DIR = os.environ.get(
    "PRACTICE9_CUSTOM_USD_DIR",
    f"{DATA_ROOT}/datasets/practice9/custom_robot/usd",
)

# This order is the contract shared by actions, observations, retarget output, and NPZ metadata.
CUSTOM_HUMANOID_30DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_yaw_joint",
    "left_ankle_pitch_joint",
    "left_foot_roll",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_yaw_joint",
    "right_ankle_pitch_joint",
    "right_foot_roll",
    "waist_yaw",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_pitch",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch",
    "neck",
    "neck_linkage_roll",
    "head_pitch",
]

CUSTOM_HUMANOID_TRACKING_BODY_NAMES = [
    "base_link",
    "left_thigh",
    "left_mid_leg",
    "left_foot",
    "right_thigh",
    "right_mid_leg",
    "right_foot",
    "chest",
    "left_upper_arm",
    "left_force_arm",
    "left_wrist",
    "right_upper_arm",
    "right_force_arm",
    "right_wrist",
]

LEG_MAJOR_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_yaw_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_yaw_joint",
]
FOOT_JOINTS = [
    "left_ankle_pitch_joint",
    "left_foot_roll",
    "right_ankle_pitch_joint",
    "right_foot_roll",
]
WAIST_JOINTS = ["waist_yaw", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_pitch",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch",
]
NECK_JOINTS = ["neck", "neck_linkage_roll", "head_pitch"]


CUSTOM_HUMANOID_30DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=CUSTOM_HUMANOID_URDF,
        usd_dir=CUSTOM_HUMANOID_USD_DIR,
        usd_file_name="urdf0711_training_30dof_tw123_yaw_waist_v3.usd",
        fix_base=False,
        merge_fixed_joints=True,
        collider_type="convex_hull",
        self_collision=False,
        force_usd_conversion=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.264),
        joint_pos={name: next((initial for target, _, initial in COORDINATES.values() if target == name), 0.0)
                   for name in CUSTOM_HUMANOID_30DOF_JOINT_NAMES},
        joint_vel={".*": 0.0},
    ),
    # Generated hard limits are already conservative bootstrap limits.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "leg_major": ImplicitActuatorCfg(
            joint_names_expr=LEG_MAJOR_JOINTS,
            effort_limit_sim=40.0,
            velocity_limit_sim=3.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=FOOT_JOINTS,
            effort_limit_sim=15.0,
            velocity_limit_sim=3.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.001,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINTS,
            effort_limit_sim=20.0,
            velocity_limit_sim=2.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINTS,
            effort_limit_sim=8.0,
            velocity_limit_sim=2.0,
            stiffness=5.0,
            damping=0.25,
            armature=0.0005,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=NECK_JOINTS,
            effort_limit_sim=3.0,
            velocity_limit_sim=2.0,
            stiffness=3.0,
            damping=0.15,
            armature=0.0002,
        ),
    },
    joint_sdk_names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
)

# Bootstrap action scales use the same 0.25 * effort / stiffness rule as G1 mimic.
CUSTOM_HUMANOID_ACTION_SCALE = {
    **{name: 0.50 for name in LEG_MAJOR_JOINTS},
    **{name: 0.375 for name in FOOT_JOINTS},
    **{name: 0.50 for name in WAIST_JOINTS},
    **{name: 0.40 for name in ARM_JOINTS},
    **{name: 0.25 for name in NECK_JOINTS},
}
