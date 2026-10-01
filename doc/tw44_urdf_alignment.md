# TW-44: Old/new URDF joint alignment

## Server project

- Training repository: `/root/gpufree-data/projects/HUMANOID_TW`
- GitHub: https://github.com/weiguang52/HUMANOID_TW
- Old 28-DoF IK model: `/root/gpufree-data/projects/IK_V2/data/urdf/Assembly.urdf`
- New 32-DoF CAD: `/root/gpufree-data/projects/urdf0711/urdf/urdf0711.urdf`
- Training 30-DoF model: `/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf`

## Generate the old joint interface

Run on the server from the repository in the `env_isaaclab` environment:

```bash
python scripts/practice9/align_custom_robot_urdf.py
```

The output is `urdf0711_old_joint_coordinates_30dof.urdf` and a sibling
`.alignment.json` under the training URDF directory. Neither input nor the
current training model is overwritten. Override `--source`, `--reference`,
and `--output` as needed. Using the raw CAD as source retains active hand rolls.

The coordinate contract is `q_new = sign * q_old + offset`. The generator
renames joints, multiplies axes by sign, incorporates offset into the joint
origin rotation, and transforms position limits (sorting reversed bounds).
It preserves source link names, geometry, inertial data, collision geometry,
effort and velocity limits. The two additional foot rolls retain their names.
Shoulder offsets are essential; renaming joints and changing axes alone is wrong.

Current RL and retargeting continue to use the new 30-DoF interface. Their
shared mapping is now in `joint_alignment.py`; `map_28_to_30` is unchanged.
The aligned output is for consumers requiring old joint coordinates. Switching
RL to this output requires matching joint configuration, action coordinates,
limits, a separate USD cache and regenerated motion FK. Existing NPZ files and
policies cannot be used unchanged.

## Mapping

Signs and offsets below convert old angles to original new model angles,
in radians. Extra foot rolls are zero in mapped old motions.

| New joint | Old joint | sign | offset (rad) |
|---|---|---:|---:|
| left_hip_linkage_pitch | left_hip_pitch_joint | -1 | 0 |
| left_thigh_roll | left_hip_roll_joint | 1 | 0 |
| left_knee_linkage_yaw | left_hip_yaw_joint | 1 | 0 |
| left_mid_leg_pitch | left_knee_pitch_joint | -1 | 0 |
| left_calf_yaw | left_ankle_yaw_joint | -1 | 0 |
| left_ankle_pitch | left_ankle_pitch_joint | 1 | 0 |
| left_foot_roll | extra DoF | 0 | 0 |
| right_hip_linkage_pitch | right_hip_pitch_joint | -1 | 0 |
| right_thigh_roll | right_hip_roll_joint | 1 | 0 |
| right_knee_linkage_yaw | right_hip_yaw_joint | 1 | 0 |
| right_mid_leg_pitch | right_knee_pitch_joint | 1 | 0 |
| right_calf_yaw | right_ankle_yaw_joint | 1 | 0 |
| right_ankle_pitch | right_ankle_pitch_joint | 1 | 0 |
| right_foot_roll | extra DoF | 0 | 0 |
| waist_yaw | waist_yaw_joint | 1 | 0 |
| gearbox_roll | waist_roll_joint | 1 | 0 |
| chest_pitch | waist_pitch_joint | -1 | 0 |
| left_shoulder_linkage_pitch | left_shoulder_pitch_joint | -1 | 0 |
| left_upper_arm_roll | left_shoulder_roll_joint | 1 | 1.57079633 |
| left_elbow_linkage_pitch | left_shoulder_yaw_joint | 1 | 0 |
| left_force_arm_yaw | left_elbow_pitch_joint | 1 | 0 |
| left_wrist_pitch | left_wrist_yaw_joint | -1 | 0 |
| right_shoulder_linkage_pitch | right_shoulder_pitch_joint | -1 | 0 |
| right_upper_arm_roll | right_shoulder_roll_joint | 1 | -1.57079633 |
| right_elbow_linkage_pitch | right_shoulder_yaw_joint | 1 | 0 |
| right_force_arm_yaw | right_elbow_pitch_joint | 1 | 0 |
| right_wrist_pitch | right_wrist_yaw_joint | -1 | 0 |
| neck | neck_yaw_joint | 1 | 0 |
| neck_linkage_roll | neck_roll_joint | 1 | 0 |
| head_pitch | neck_pitch_joint | -1 | 0 |

## Validation and limitations

All 28 mapped axes agree after parent-chain rotations and shoulder offsets
are applied in the root frame. The report records input SHA-256 hashes and
per-joint results. At baseline plus 32 deterministic arbitrary joint poses,
all link transforms match the source new robot (maximum matrix error about
2.78e-16). Tests cover real-model FK, 30-DoF count, limit conversion, unchanged
effort/velocity, wrong-axis rejection, overwrite rejection and permuted motion columns.
Remote validation: `python -m pytest -q tests/practice9` passed 71 tests and
8 subtests. No model assets were copied to the local computer.

Old/new dimensions and joint centers differ. The old waist is yaw-pitch-roll;
the new waist is yaw-roll-pitch. Alignment of names, positive directions and
reference zero does not make FK of the two different robots identical at all
poses. Old IK solutions remain bootstrap retargeting, rather than exact IK
solutions for the new robot. Training limits and dynamics remain bootstrap values.
