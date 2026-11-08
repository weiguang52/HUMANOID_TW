# Issue #5: factor attribution and multi-training-seed replication

Issue: https://github.com/weiguang52/HUMANOID_TW/issues/5 . Defaults unchanged. All work and artifacts remain on the remote data disk.

## Protocol

Existing robust08 seed1 and its seed2 replication use clean, push-only, friction/mass-only, fixed20ms-delay-only, observation-noise-only and original combined challenge. Every condition uses3 evaluation seeds,128 environments and20s per command at0.3/0.5/0.8m/s. Per-command reset seed is evaluation_seed*1000+command_index. Non-target perturbations are disabled. Combined retains random0/20ms delay; isolated delay is fixed20ms. These are new paired streams, not an exact replay of historical unpaired random streams.

## Single-factor fall counts

Each cell lists evaluation seeds1/2/3; each denominator is128.

| Condition | Command | Training seed1 | Training seed2 | Mean fall rate across training seeds | Sample SD across training seeds |
|---|---|---|---|---:|---:|
| clean | forward_03 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| clean | forward_05 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| clean | forward_08 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| push | forward_03 | 1/1/0 | 1/4/1 | 1.042% | 0.737% |
| push | forward_05 | 3/5/6 | 3/2/7 | 3.385% | 0.368% |
| push | forward_08 | 5/7/8 | 6/9/7 | 5.469% | 0.368% |
| material | forward_03 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| material | forward_05 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| material | forward_08 | 1/1/1 | 3/6/9 | 2.734% | 2.762% |
| delay | forward_03 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| delay | forward_05 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| delay | forward_08 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| noise | forward_03 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| noise | forward_05 | 0/0/0 | 0/0/0 | 0.000% | 0.000% |
| noise | forward_08 | 0/0/0 | 1/0/0 | 0.130% | 0.184% |
| combined | forward_03 | 3/2/3 | 8/4/5 | 3.255% | 1.657% |
| combined | forward_05 | 9/12/17 | 11/15/16 | 10.417% | 0.737% |
| combined | forward_08 | 28/28/30 | 37/34/50 | 26.953% | 6.445% |

The identical frozen attribution rule yields {'1': None, '2': None} across training seeds. See attribution_training_seed_replication.json. This replication does not change the already frozen training selection.

## Attribution and limits

no single dominant factor; strongest isolated stressor is exploratory only. The predeclared rule and full seed1 scores are in factor_attribution.json and frozen_plan.md. The training factor is push. If no dominant factor meets the frozen criteria, the training ablations are exploratory probes of the strongest isolated stressor, not proof of its causal dominance.

Only two training seeds are available, both conditional on the identical seed1 robust05 parent and full optimizer. The sample SD above is descriptive and does not establish broad training-seed robustness. A combined failure rate higher than isolated conditions is consistent with interacting stresses, but these runs do not identify a unique interaction mechanism.

## Ablation outcomes

Remove disables only the selected component. Strengthen changes only its strength relative to the original combined robust08 package. All five formal runs use4096 environments,1500 logged updates and the exact common parent; no std reset or checkpoint selection. See training_plan.json and saved configurations.

| Model | Condition | 0.3m/s falls | 0.5m/s falls | 0.8m/s falls |
|---|---|---|---|---|
| robust08_s1 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| robust08_s1 | push | 1/1/0 | 3/5/6 | 5/7/8 |
| robust08_s1 | material | 0/0/0 | 0/0/0 | 1/1/1 |
| robust08_s1 | delay | 0/0/0 | 0/0/0 | 0/0/0 |
| robust08_s1 | noise | 0/0/0 | 0/0/0 | 0/0/0 |
| robust08_s1 | combined | 3/2/3 | 9/12/17 | 28/28/30 |
| robust08_s2 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| robust08_s2 | push | 1/4/1 | 3/2/7 | 6/9/7 |
| robust08_s2 | material | 0/0/0 | 0/0/0 | 3/6/9 |
| robust08_s2 | delay | 0/0/0 | 0/0/0 | 0/0/0 |
| robust08_s2 | noise | 0/0/0 | 0/0/0 | 1/0/0 |
| robust08_s2 | combined | 8/4/5 | 11/15/16 | 37/34/50 |
| remove_s1 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| remove_s1 | push | 54/42/57 | 74/70/66 | 88/77/86 |
| remove_s1 | combined | 48/61/55 | 69/89/78 | 91/93/95 |
| remove_s2 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| remove_s2 | push | 40/42/39 | 62/56/49 | 80/75/78 |
| remove_s2 | combined | 52/50/50 | 64/71/64 | 87/95/90 |
| strengthen_s1 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| strengthen_s1 | push | 0/0/1 | 1/0/0 | 1/3/5 |
| strengthen_s1 | combined | 2/1/2 | 5/5/6 | 19/13/21 |
| strengthen_s2 | clean | 0/0/0 | 0/0/0 | 0/0/0 |
| strengthen_s2 | push | 1/0/0 | 0/2/1 | 8/12/6 |
| strengthen_s2 | combined | 1/1/2 | 6/5/8 | 25/21/19 |

## Acceptance

Combined-challenge target: every evaluation seed at0.3 and0.5m/s must have fall rate strictly below1%;0.8m/s is recorded without threshold changes.

- robust08_s1: FAIL;0.8m/s counts [28, 28, 30]
- robust08_s2: FAIL;0.8m/s counts [37, 34, 50]
- remove_s1: FAIL;0.8m/s counts [91, 93, 95]
- remove_s2: FAIL;0.8m/s counts [87, 95, 90]
- strengthen_s1: FAIL;0.8m/s counts [19, 13, 21]
- strengthen_s2: FAIL;0.8m/s counts [25, 21, 19]
- remove, both training seeds: FAIL
- strengthen, both training seeds: FAIL

## Matched-training-seed interpretation

Neither training seed meets the frozen single-dominant-factor rule. Push is the strongest isolated stressor, so the selected training ablations remain exploratory. Removing push exposure worsens combined-challenge falls at all three tested speeds in both training seeds. Strengthening only push exposure reduces the counts at all three speeds in both training seeds, but neither seed passes the low/mid-speed target. These are descriptive comparisons conditional on one shared parent; two seeds and this isolated-factor design do not establish a unique interaction mechanism.

The following pooled counts sum three evaluation seeds (denominator384), without replacing the stricter per-evaluation-seed acceptance.

| Training seed | Command | Original combined | Remove push combined | Strengthen push combined |
|---|---|---:|---:|---:|
| 1 | forward_03 | 8/384 | 164/384 | 5/384 |
| 1 | forward_05 | 38/384 | 236/384 | 16/384 |
| 1 | forward_08 | 86/384 | 279/384 | 53/384 |
| 2 | forward_03 | 17/384 | 152/384 | 4/384 |
| 2 | forward_05 | 42/384 | 199/384 | 19/384 |
| 2 | forward_08 | 121/384 | 272/384 | 65/384 |

No default policy/configuration is promoted by this research delivery. A completed experiment is not equivalent to performance acceptance.

## Physical mass audit

The original source base mass is0.07990783598kg. Isaac Lab clamps randomized masses to1e-6kg and rescales inertia. Some material conditions include extremely light bodies; inspect physics_diagnostics in the recorded JSON. These are part of the original package and must not be interpreted as realistic hardware identification. No negative mass is applied. Mass diagnostics were added after the earliest clean/push runs; absent diagnostic fields are not fabricated.

## Worst-command colored front/side videos

The worst instruction per model is selected by total combined-challenge failures over all3 evaluation seeds, ties resolved in0.3/0.5/0.8 order. Videos are diagnostic one-environment seed1 replays, with failure/reset statistics in their JSON; they do not replace128-env acceptance. Camera follows translation and heading. Exact precolored USD is checked against original composed attributes and non-material relationships; no runtime material binding. Velocity commands have no HumanML3D source NPY.

- [remove_s1_worst_front_forward_08_front](videos/remove_s1_worst_front_forward_08_front.mp4)
- [remove_s1_worst_side_forward_08_side](videos/remove_s1_worst_side_forward_08_side.mp4)
- [remove_s2_worst_front_forward_08_front](videos/remove_s2_worst_front_forward_08_front.mp4)
- [remove_s2_worst_side_forward_08_side](videos/remove_s2_worst_side_forward_08_side.mp4)
- [robust08_s1_worst_front_forward_08_front](videos/robust08_s1_worst_front_forward_08_front.mp4)
- [robust08_s1_worst_side_forward_08_side](videos/robust08_s1_worst_side_forward_08_side.mp4)
- [robust08_s2_worst_front_forward_08_front](videos/robust08_s2_worst_front_forward_08_front.mp4)
- [robust08_s2_worst_side_forward_08_side](videos/robust08_s2_worst_side_forward_08_side.mp4)
- [strengthen_s1_worst_front_forward_08_front](videos/strengthen_s1_worst_front_forward_08_front.mp4)
- [strengthen_s1_worst_side_forward_08_side](videos/strengthen_s1_worst_side_forward_08_side.mp4)
- [strengthen_s2_worst_front_forward_08_front](videos/strengthen_s2_worst_front_forward_08_front.mp4)
- [strengthen_s2_worst_side_forward_08_side](videos/strengthen_s2_worst_side_forward_08_side.mp4)

Joint color legend: [joint_colors.json](joint_colors.json). Numerical and visual validation records accompany the report. Archived logs remove ANSI decoration and trailing whitespace; raw originals remain on the remote data disk. Failed/intermediate debug outputs remain excluded on the remote data disk.

## Project tracking

Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . GitHub issue replies record the measured result and any unmet gate. SuperMimic remains stopped; this is a separate stage1 velocity-control attribution task.
