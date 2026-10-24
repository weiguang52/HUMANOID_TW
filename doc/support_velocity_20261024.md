# Stationary support velocity ablation

This is an opt-in experiment, not a validated improvement. Prior TW202 precision
rewards regressed standing completion (6/6 to 4/6); do not repeat that change.
HeadLocked model_27995 remains the baseline asset and starting checkpoint.

Method sources inspected 2026-10-24:
- https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py
- https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/rewards/tracking.py
- https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/rewards/regularization.py

Use global velocity tracking plus reference contact eligibility. No upstream code
is copied. Keep existing position/pose/contact rewards, observations, limits,
head locks, data, and reference timing. Add only a bounded pelvis XY velocity
tracking cost (weight -0.25, scale 0.02 m/s). Eligibility requires standing_upper,
two known support labels, and smooth reference speed/yaw-rate gates. The gate
uses reference rather than actual contact, so actual loss of support cannot
remove the cost. Low root speed alone does not classify a treadmill as standing.

Data: cpp_refresh_20261024/train.json, 1348 motions / 788767 frames. Of 634807
standing frames, 593487 have nonzero eligibility; mean standing gate is 0.901918.
These counts inspect existing labels; no labels or trajectories were edited.

Both runs start from model_27995.pt, seed42, 1024 environments, 2000 additional
iterations, same PPO settings. Baseline keeps HeadLocked; support adds the cost.
Use separate GPUs and record that hardware nondeterminism remains possible.
18 tensor/regression tests passed. Both 64-environment / 2-iteration smoke runs
completed, with the new reward present only in the candidate configuration.

Launch in tmux via run_support_experiment.sh GPU baseline|support after running
train_support_ablation.sh GPU VARIANT smoke. prepare_support_evaluation.py
creates isolated jobs from the prior validated colored evaluation protocol;
template hashes are saved. Original reports and checkpoints are preserved.

Acceptance: compare 42 held-out replays and equal-duration 20-second static
cases, including early terminations and 5-second windows. Seek lower drift
without loss of walking/standing/turn completion or increased wrist error.
Do not infer an improvement from training reward or unequal surviving durations.
Nonplanar clips remain terrain-mismatch diagnostics. One training seed is not
proof of generality. Candidate is not promoted automatically.

Both runs automatically generate colored front/side robot videos plus original
HumanML3D references. Colors use the existing separately baked HeadLocked USD;
physics and locked joints match the training model. Front camera follows
translation, not heading. Raw human timing is not phase aligned to robot replay.
State: /root/gpufree-data/datasets/practice9/support_ablation_20261024
Archive: validation_artifacts/support_ablation_20261024/{baseline,support}/evaluation_colored/neck_only
