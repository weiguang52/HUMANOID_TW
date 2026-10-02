"""TW-44 table: aligned angle = sign * original CAD angle + initial reading.

The pre-change training initial readings were all zero. Offsets are readings
at that same physical pose, not additional physical rotations.
"""
import math

CONTRACT_VERSION = 'tw44_table_v2'
LIMIT_OVERRIDES = {name: (-math.pi / 2, math.pi / 2) for name in (
    'left_knee_pitch_joint', 'right_knee_pitch_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint')}

COORDINATES = {
    'left_hip_linkage_pitch': ('left_hip_pitch_joint', -1, 0.0),
    'left_thigh_roll': ('left_hip_roll_joint', 1, 0.0),
    'left_knee_linkage_yaw': ('left_hip_yaw_joint', 1, 0.0),
    'left_mid_leg_pitch': ('left_knee_pitch_joint', -1, 0.0),
    'left_calf_yaw': ('left_ankle_yaw_joint', -1, 0.0),
    'left_ankle_pitch': ('left_ankle_pitch_joint', 1, 0.0),
    'right_hip_linkage_pitch': ('right_hip_pitch_joint', -1, 0.0),
    'right_thigh_roll': ('right_hip_roll_joint', 1, 0.0),
    'right_knee_linkage_yaw': ('right_hip_yaw_joint', 1, 0.0),
    'right_mid_leg_pitch': ('right_knee_pitch_joint', 1, 0.0),
    'right_calf_yaw': ('right_ankle_yaw_joint', 1, 0.0),
    'right_ankle_pitch': ('right_ankle_pitch_joint', 1, 0.0),
    'left_shoulder_linkage_pitch': ('left_shoulder_pitch_joint', -1, 0.0),
    'left_upper_arm_roll': ('left_shoulder_roll_joint', 1, -math.pi / 2),
    'left_elbow_linkage_pitch': ('left_shoulder_yaw_joint', 1, 0.0),
    'left_force_arm_yaw': ('left_elbow_pitch_joint', 1, 0.0),
    'right_shoulder_linkage_pitch': ('right_shoulder_pitch_joint', -1, 0.0),
    'right_upper_arm_roll': ('right_shoulder_roll_joint', 1, math.pi / 2),
    'right_elbow_linkage_pitch': ('right_shoulder_yaw_joint', 1, 0.0),
    'right_force_arm_yaw': ('right_elbow_pitch_joint', 1, 0.0),
}


def rename(name):
    return COORDINATES.get(name, (name, 1, 0.0))[0]


def convert_motion(names, positions, velocities, requested_names, contract=None):
    """Convert legacy CAD columns only when aligned coordinates are requested.

Body FK arrays remain valid because the conversion preserves physical pose.
Current-version aligned clips and other robot families pass through unchanged.
"""
    if names is None or requested_names is None:
        return names, positions, velocities
    legacy = [name for name in COORDINATES if name in names and rename(name) in requested_names]
    if not legacy:
        if all(new in names for new, _, _ in COORDINATES.values()) and contract != CONTRACT_VERSION:
            raise ValueError('Stale aligned motion coordinates; regenerate with ' + CONTRACT_VERSION)
        return names, positions, velocities
    converted_names = [rename(name) for name in names]
    if len(set(converted_names)) != len(converted_names):
        raise ValueError('Mixed legacy/aligned joint coordinate names')
    positions = positions.copy()
    velocities = velocities.copy()
    for name in legacy:
        column = names.index(name)
        _, sign, offset = COORDINATES[name]
        positions[:, column] = sign * positions[:, column] + offset
        velocities[:, column] *= sign
    return converted_names, positions, velocities
