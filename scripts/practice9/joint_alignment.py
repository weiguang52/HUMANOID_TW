"""Shared old IK to new CAD joint coordinate contract."""

import math

CUSTOM_JOINT_NAMES = [
    "left_hip_linkage_pitch", "left_thigh_roll", "left_knee_linkage_yaw",
    "left_mid_leg_pitch", "left_calf_yaw", "left_ankle_pitch", "left_foot_roll",
    "right_hip_linkage_pitch", "right_thigh_roll", "right_knee_linkage_yaw",
    "right_mid_leg_pitch", "right_calf_yaw", "right_ankle_pitch", "right_foot_roll",
    "waist_yaw", "gearbox_roll", "chest_pitch",
    "left_shoulder_linkage_pitch", "left_upper_arm_roll", "left_elbow_linkage_pitch",
    "left_force_arm_yaw", "left_wrist_pitch",
    "right_shoulder_linkage_pitch", "right_upper_arm_roll", "right_elbow_linkage_pitch",
    "right_force_arm_yaw", "right_wrist_pitch",
    "neck", "neck_linkage_roll", "head_pitch",
]

# new_name: (old_name, sign, zero offset)
MAPPING = {
    "left_hip_linkage_pitch": ("left_hip_pitch_joint", -1.0, 0.0),
    "left_thigh_roll": ("left_hip_roll_joint", 1.0, 0.0),
    "left_knee_linkage_yaw": ("left_hip_yaw_joint", 1.0, 0.0),
    "left_mid_leg_pitch": ("left_knee_pitch_joint", -1.0, 0.0),
    "left_calf_yaw": ("left_ankle_yaw_joint", -1.0, 0.0),
    "left_ankle_pitch": ("left_ankle_pitch_joint", 1.0, 0.0),
    "left_foot_roll": (None, 0.0, 0.0),
    "right_hip_linkage_pitch": ("right_hip_pitch_joint", -1.0, 0.0),
    "right_thigh_roll": ("right_hip_roll_joint", 1.0, 0.0),
    "right_knee_linkage_yaw": ("right_hip_yaw_joint", 1.0, 0.0),
    "right_mid_leg_pitch": ("right_knee_pitch_joint", 1.0, 0.0),
    "right_calf_yaw": ("right_ankle_yaw_joint", 1.0, 0.0),
    "right_ankle_pitch": ("right_ankle_pitch_joint", 1.0, 0.0),
    "right_foot_roll": (None, 0.0, 0.0),
    "waist_yaw": ("waist_yaw_joint", 1.0, 0.0),
    "gearbox_roll": ("waist_roll_joint", 1.0, 0.0),
    "chest_pitch": ("waist_pitch_joint", -1.0, 0.0),
    "left_shoulder_linkage_pitch": ("left_shoulder_pitch_joint", -1.0, 0.0),
    "left_upper_arm_roll": ("left_shoulder_roll_joint", 1.0, math.pi / 2.0),
    "left_elbow_linkage_pitch": ("left_shoulder_yaw_joint", 1.0, 0.0),
    "left_force_arm_yaw": ("left_elbow_pitch_joint", 1.0, 0.0),
    "left_wrist_pitch": ("left_wrist_yaw_joint", -1.0, 0.0),
    "right_shoulder_linkage_pitch": ("right_shoulder_pitch_joint", -1.0, 0.0),
    "right_upper_arm_roll": ("right_shoulder_roll_joint", 1.0, -math.pi / 2.0),
    "right_elbow_linkage_pitch": ("right_shoulder_yaw_joint", 1.0, 0.0),
    "right_force_arm_yaw": ("right_elbow_pitch_joint", 1.0, 0.0),
    "right_wrist_pitch": ("right_wrist_yaw_joint", -1.0, 0.0),
    "neck": ("neck_yaw_joint", 1.0, 0.0),
    "neck_linkage_roll": ("neck_roll_joint", 1.0, 0.0),
    "head_pitch": ("neck_pitch_joint", -1.0, 0.0),
}
