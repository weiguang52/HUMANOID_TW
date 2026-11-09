# Issue #6: bounded mass randomization and strengthened-push retraining

Issue: https://github.com/weiguang52/HUMANOID_TW/issues/6 . Defaults remain legacy.

## Mass correction and 4096-environment audit

Only base_link is randomized. Original robust08 adds U(-0.08,0.15)kg to nominal0.07990783453kg and safety-clamps at1e-6kg. The opt-in bounded profile multiplies nominal by U(0.5,1.5), giving theoretical base range0.0399539173..0.1198617518kg. Same startup timing and inertia scaling. Static friction remains[0.4,1.4], dynamic[0.3,1.2]. Other body properties remain unchanged. This bounded research prior is not hardware identification or mean/variance-matched to the original additive distribution.

Every audit uses4096 actual PhysX environments, all31 bodies. Requested masses are captured within the real mass event immediately before safety clamping, then actual applied masses are read back. Full per-body min/max/mean/quantiles, <1% nominal fractions, actual clamp counts and requested-clamp counts are saved in four mass_audit JSONs and mass_distribution_table.csv.

| Profile | Seed | Base min kg | Base mean kg | Base max kg | Below1% nominal | At safety clamp | Requested at/below clamp |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 1 | 9.99999997e-07 | 0.115697287 | 0.22984001 | 20/4096 (0.4883%) | 3 | 3 |
| legacy | 2 | 5.96791506e-06 | 0.114247538 | 0.22989288 | 19/4096 (0.4639%) | 0 | 0 |
| bounded | 1 | 0.0399624296 | 0.0801820979 | 0.119838186 | 0/4096 (0.0000%) | 0 | 0 |
| bounded | 2 | 0.0399880111 | 0.0796784237 | 0.119856559 | 0/4096 (0.0000%) | 0 | 0 |

The installed Isaac Lab sampler passed four unit tests: lower endpoint and partial reset without accumulation; uniform bounds and untouched non-target bodies; old negative draws hit the safety clamp; legacy/default and disabled event behavior stay unchanged. Smoke64-env2-update resume also passed. Initial audit adapter signature failure was corrected before any formal training; its debug log remains remote and is excluded.

## Protocol and controlled comparisons

Both new models use4096 environments and1500 updates from the exact shared robust05 parent, SHA256 c200e355d29fda60225761550ea80642bb222778ce50c00ae3deb3d95fa7a38d, with full optimizer resume and no std reset or checkpoint selection. Training push is +/-0.45m/s. Both seeds share one seed1 parent; they are not full-pipeline independent replications.

Evaluation: deterministic policy,128 environments,20s per0.3/0.5/0.8m/s command; eval seeds1/2/3, paired reset seed=eval_seed*1000+command_index. Combined evaluation retains original +/-0.3m/s pushes, random0/20ms delay and noise. Material-only disables these non-target factors. New mass distribution is used in both conditions.24 fresh quantitative jobs, plus archived legacy references.

To separate changed evaluation physics from retraining, historical robust08 checkpoints are also evaluated with bounded material; historical strengthened-push checkpoints with bounded combined. Paired reset seeds do not imply identical trajectories or identical startup physics across changed distributions.

Each cell below lists eval seeds1/2/3, denominator128 each.

| Model | Evaluated mass | Condition | 0.3m/s | 0.5m/s | 0.8m/s |
|---|---|---|---|---|---|
| robust08_s1 | legacy | material | 0/0/0 | 0/0/0 | 1/1/1 |
| old_robust08_s1 | bounded | material | 0/0/0 | 0/0/0 | 0/1/3 |
| strengthen_s1 | legacy | combined | 2/1/2 | 5/5/6 | 19/13/21 |
| old_strengthen_s1 | bounded | combined | 1/4/1 | 4/2/7 | 16/12/14 |
| bounded_strengthen_s1 | bounded | combined | 3/2/1 | 3/7/8 | 11/14/21 |
| bounded_strengthen_s1 | bounded | material | 0/0/0 | 0/1/3 | 0/0/3 |
| robust08_s2 | legacy | material | 0/0/0 | 0/0/0 | 3/6/9 |
| old_robust08_s2 | bounded | material | 0/0/0 | 0/0/0 | 4/4/7 |
| strengthen_s2 | legacy | combined | 1/1/2 | 6/5/8 | 25/21/19 |
| old_strengthen_s2 | bounded | combined | 2/0/1 | 8/6/5 | 22/20/21 |
| bounded_strengthen_s2 | bounded | combined | 1/3/2 | 4/9/7 | 25/23/29 |
| bounded_strengthen_s2 | bounded | material | 0/0/0 | 0/0/0 | 0/0/0 |

## Separating evaluation-physics correction from retraining

Each pooled count below sums three evaluation seeds (denominator384 per speed). The mass-only comparison uses the same robust08 policy before/after the evaluation correction. The combined comparison first holds the old strengthened policy fixed while changing evaluation mass, then holds bounded evaluation fixed while changing the trained policy.

| Seed | Command | Robust08 material legacy -> bounded | Old strengthened combined legacy -> bounded | Bounded-trained combined |
|---|---|---|---|---|
| 1 | forward_03 | 0 -> 0 | 5 -> 6 | 6 |
| 1 | forward_05 | 0 -> 0 | 16 -> 13 | 18 |
| 1 | forward_08 | 3 -> 4 | 53 -> 42 | 46 |
| 2 | forward_03 | 0 -> 0 | 4 -> 3 | 6 |
| 2 | forward_05 | 0 -> 0 | 19 -> 19 | 20 |
| 2 | forward_08 | 18 -> 15 | 65 -> 63 | 77 |

- Robust08 training seed1: material-only pooled fall rate 0.2604% -> 0.3472% (same policy, changed mass distribution).
- Robust08 training seed2: material-only pooled fall rate 1.5625% -> 1.3021% (same policy, changed mass distribution).

## Acceptance and interpretation

Frozen target: every evaluation seed at0.3 and0.5m/s strictly<1%, with0.8m/s recorded.
- bounded_strengthen_s1: FAIL;0.8m/s falls [11, 14, 21]
- bounded_strengthen_s2: FAIL;0.8m/s falls [25, 23, 29]

Both training seeds: FAIL. No default configuration/policy is promoted. Completed fixed-budget experiments do not imply a passed performance gate.

Crossed comparisons are descriptive. Correcting mass changes mean, variance and tails, so differences cannot be attributed solely to removing safety-clamped samples. New-vs-old trained-policy comparisons under the same bounded evaluation reduce that confounding but two conditional training seeds do not prove broad robustness. The original #5 no-single-dominant-factor conclusion is not rewritten by these new protocols.

Measured conclusion: the sampling defect is fixed (no clamp/near-zero masses), but this fixed-budget retraining does not improve falls under the common bounded combined evaluation: five of six training-seed/speed pooled counts are higher than the old strengthened policies, and one is equal. Material-only correction with fixed robust08 policies changes rates in opposite directions across training seeds. No consistent policy-robustness benefit is demonstrated.


Descriptive two-training-seed means and sample standard deviations are saved in training_seed_variance.json; they are not confidence guarantees.

## Colored worst-command front/side videos

Worst command is selected from summed combined failures over three evaluation seeds, ties0.3/0.5/0.8. Each video is a one-environment diagnostic replay, not128-env acceptance. Camera follows translation and heading; front/side labels are in filenames. Exact precolored USD is compared with original physics and non-material relationships; no dynamic material binding. Velocity commands have no HumanML3D source NPY. Joint color legend: joint_colors.json.
- [bounded_strengthen_s1_worst_front_forward_08_front](videos/bounded_strengthen_s1_worst_front_forward_08_front.mp4)
- [bounded_strengthen_s1_worst_side_forward_08_side](videos/bounded_strengthen_s1_worst_side_forward_08_side.mp4)
- [bounded_strengthen_s2_worst_front_forward_08_front](videos/bounded_strengthen_s2_worst_front_forward_08_front.mp4)
- [bounded_strengthen_s2_worst_side_forward_08_side](videos/bounded_strengthen_s2_worst_side_forward_08_side.mp4)

All artifacts were generated remotely. Archived logs strip ANSI decoration/trailing whitespace; raw originals and debug outputs remain on the remote data disk. No mounted file is copied to local storage.

## Project tracking

Notion replaces Linear: https://www.notion.so/403325d8231d8172995bf19510b8d760 . GitHub issue reply records measured results and any unmet gate. SuperMimic remains stopped; this is stage1 velocity-control research.
