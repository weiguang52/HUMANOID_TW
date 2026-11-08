# TW stage1 FT5 training result

Source run: `logs/rsl_rl/tw_stage1/2026-11-07_17-12-49_tw_stage1_s1_ft5`. Final checkpoint: `model_5997.pt`.

Archived final model, run configuration, source diff and original TensorBoard event log. Intermediate checkpoints remain in the original run directory.

## Training metrics

| Metric | Last step | Last value | Last 100 mean |
|---|---:|---:|---:|
| Episode_Reward/track_lin_vel_xy | 5997 | 2.86629 | 2.83744 |
| Episode_Reward/track_ang_vel_z | 5997 | 0.463816 | 0.458896 |
| Episode_Reward/alive | 5997 | 0.495691 | 0.492197 |
| Episode_Reward/base_linear_velocity | 5997 | -0.00174638 | -0.00176658 |
| Episode_Reward/base_angular_velocity | 5997 | -0.0758716 | -0.0780961 |
| Episode_Reward/flat_orientation_l2 | 5997 | -0.00855887 | -0.00852687 |
| Episode_Reward/base_height | 5997 | -8.09082e-05 | -9.21351e-05 |
| Episode_Reward/joint_acc | 5997 | -0.00228516 | -0.00232906 |
| Episode_Reward/action_rate | 5997 | -0.0140156 | -0.0142141 |
| Episode_Reward/dof_pos_limits | 5997 | -1.28427e-05 | -1.3028e-05 |
| Episode_Reward/joint_deviation_arms | 5997 | -0.0355618 | -0.0356749 |
| Episode_Reward/joint_deviation_waist | 5997 | -0.02974 | -0.0300713 |
| Episode_Reward/joint_deviation_neck | 5997 | -0.015457 | -0.0150899 |
| Episode_Reward/joint_deviation_hips | 5997 | -0.070233 | -0.0710744 |
| Episode_Reward/gait | 5997 | 0.430197 | 0.455129 |
| Episode_Reward/swing_height | 5997 | 0.162331 | 0.16749 |
| Episode_Reward/feet_slide | 5997 | -0.00522705 | -0.00526711 |
| Episode_Reward/undesired_contacts | 5997 | -8.33333e-06 | -3.08245e-05 |
| Curriculum/lin_vel_cmd_levels | 5997 | 0.3 | 0.3 |
| Metrics/base_velocity/error_vel_xy | 5997 | 0.0641188 | 0.0665193 |
| Metrics/base_velocity/error_vel_yaw | 5997 | 0.204014 | 0.209485 |
| Episode_Termination/time_out | 5997 | 0.963247 | 0.965278 |
| Episode_Termination/base_height | 5997 | 0.0367533 | 0.0347225 |
| Episode_Termination/base_contact | 5997 | 0 | 0 |
| Loss/value | 5997 | 0.0329203 | 0.0316439 |
| Loss/surrogate | 5997 | -0.00148805 | -0.00208127 |
| Loss/entropy | 5997 | -71.154 | -70.5937 |
| Loss/learning_rate | 5997 | 3.375e-05 | 2.67437e-05 |
| Policy/mean_std | 5997 | 0.0349594 | 0.0355165 |
| Perf/total_fps | 5997 | 92648 | 93332.4 |
| Perf/collection_time | 5997 | 0.959597 | 0.95971 |
| Perf/learning_time | 5997 | 0.10145 | 0.0937249 |
| Train/mean_reward | 5997 | 83.5815 | 82.8393 |
| Train/mean_episode_length | 5997 | 992.25 | 983.728 |
| Train/mean_reward/time | 1605 | 83.5815 | 82.8393 |
| Train/mean_episode_length/time | 1605 | 992.25 | 983.728 |

## Interpretation

These are training metrics only. They do not establish independent evaluation performance or deployment acceptance. No new simulation evaluation was run for this archival submission. The final checkpoint filename alone does not prove training completed its intended budget.

## SHA256

- `events.out.tfevents.1794042804.gpufree-container.84773.0`: `f855e124adec0710ecac1fbf7814fcdfb75e023c2872c9e4e14fb2bcf5684212`
- `git/HUMANOID_TW.diff`: `e16bb6514be32e0788051aedf2d72f86dc909facb4d6fbd00bf1fb1b6e9f7358`
- `metrics_summary.json`: `61c0a73fe22e2695f3b0595d91a6fd55935def9f8f465fab77910498c5bf1434`
- `model_5997.pt`: `58eccca805e50c28c501104be241b48bb77822fe40bf4e3efdda30174827577a`
- `params/agent.yaml`: `fa65585e29a147d24692a4acc72faaa19b5528df7d36ca79df398e03688a2349`
- `params/env.yaml`: `d7d88340db0b207b730c124266ff3ac45ddcf0a3091e8efe8b0ad9e9899103fe`
