# TW-44: Apply the user joint coordinate table

Training URDF: `/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf`.
Backup: sibling `backups/urdf0711_training_30dof.pre_tw44_table.urdf`.

The former training initial angles were all zero. The table's `*_hip_linkage_joint`
and `*_mid_leg_pittch` refer to actual `*_hip_linkage_pitch` and `*_mid_leg_pitch`.
Only the 20 listed joints change names. Six listed axes reverse.
Unlisted wrists, feet, waist and neck retain their original names/conventions.

For a listed joint: `q_aligned = sign * q_original + C`, where C is the table's
requested initial reading. C is -pi/2 for left_shoulder_pitch_joint, +pi/2 for
right_shoulder_roll_joint, and zero otherwise. The original initial reading B
was verified as zero in the training config. This preserves the physical pose:

- `axis_aligned = sign * axis_original`;
- `origin_aligned = origin_original * Rot(axis_original, -sign*C)`;
- position bounds become sorted `sign*bounds + C`;
- velocities multiply by sign, without adding C;
- init_state uses C, rather than putting the shifted joints at numerical zero.

The generator applies this once to a fresh raw CAD export and records the
coordinate contract plus initial readings in the metadata sidecar. The raw CAD
is unchanged. Repeated generation produces identical URDF bytes. The Isaac
config uses a versioned USD filename so an old converted asset is not reused.
Existing legacy motion clips are transformed in memory on load; body FK data
remains valid because physical poses are identical. No motion dataset is rewritten.
Old policies must not be resumed unchanged across this action-coordinate change.

Validation on the remote Linux environment:

- 65 poses (initial plus 64 random poses inside original limits), all link
  transforms preserved; maximum matrix error 5.55e-16.
- 72 tests plus 8 subtests passed, including legacy motion loading, signs,
  initial readings, transformed limits and deterministic regeneration.
- Isaac Lab headless: 2 environments, one PPO iteration completed with the
  existing locomotion manifest. Checkpoint under
  `logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d/2026-10-02_11-07-06_tw44_table_smoke`.

Run generation with `python scripts/practice9/prepare_custom_robot_urdf.py`.
