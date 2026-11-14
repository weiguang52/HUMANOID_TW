# Issue20: start phase 0/10 x original/strict symmetric USD

Outcome: **simulator_not_equivariant**. Training updates: 0. Default model/configuration unchanged.

Next issue: investigate simulator equivariance, including contact-solver ordering and finite-precision state/clock effects. No extra experiments in this issue.

## Scope and provenance

Frozen rules SHA256 `482b0eace32724500e35a7e5ff126fae6304bbceeb96ae099fe02743d34f2fac`. Exact five issue19 final sym_wrap checkpoints, deterministic policy, 1024 environments, evaluation seeds1/2/3, 20 seconds, four commands. The five training seeds are the bootstrap units; three evaluation-seed/trial means are averaged within each. Exact all3125 paired seed bootstrap resamples, individual percentile95 intervals without simultaneous-coverage correction.

Completed formal jobs: 30; commands: 120; initial trials: 122880. Original-USD30 jobs skipped as required after symmetric-USD prerequisite failure; Phi/Gamma and P are not inferred.

## Prerequisites and limits

Five102400-state phase equivalence gates passed, including actual float32 idx arithmetic. Zero-offset branch delegates the unchanged issue19 gait observation; full-size four-command1000-step actions/states and archived issue19 yaw arrays reproduce bitwise in preflight. Every original-USDk0 formal job, if reached, is also checked against all corresponding archived yaw arrays. Pure spatial reflection preserves phase; phase+10 uses idx=(episode_step+10)%20. Only gait observations shift; rewards and default configurations are unchanged.

The strict model mirrors left bodies onto right bodies, including mass, COM, tensor inertia, joint frames/limits/drives, collision transforms/meshes. Central COM y and odd inertia terms are projected to sagittal symmetry; center colliders use the left-half-clipped convex hull plus its mirror. All original actuator/default settings already satisfy reflection and remain unchanged. See usd_attribute_differences.csv for every authored difference, usd_symmetry_audit.json and runtime_symmetry_audit.json for complete numerical checks. Actual PhysX asset-layer provenance was checked.

Initial k10 root/joint/action histories are reflected from the same reset seed, including root quaternion and polar/axial velocities. Clean-protocol observation corruption is disabled, so no unmirrored random observation noise exists. No new random samples are consumed. All initial-state errors are reported per job in the analysis JSON.

Initial-episode yaw never joins reset jumps. Full20s paired S/D exclude censored pairs and their counts are explicit; common-prefix S/D and exposure are retained separately. The prerequisite requires all initial pairs to supply20s metrics to establish its full20s prediction. Half-cycle labels use the actual shifted clock: signed yaw reverses under reflection, so k10 left should equal negative k0 right. Complete20-step periods only; interrupted periods are excluded and counts retained. Paired five-seed95 swap-residual intervals must include0; no unreported numerical tolerance.

Source policy equivariance does not prove simulator/contact equivariance. Float32 sin/cos at idx+10 differs slightly from exact negation; the measured action-level error is retained. Neither a small structural difference nor a failed trajectory mirror gate alone proves a physical cause.

## Symmetric-USD trajectory prerequisite

| Command | mean S deg | 95% | abs(mean S)<=5 | left-swap95 includes0 | right-swap95 includes0 | all20s pairs |
|---|---:|---|---|---|---|---|
|stand|-19.4562|[-38.8127, -4.1503]|False|False|False|True|
|forward_03|38.0375|[0.794463, 73.3488]|False|False|False|True|
|forward_05|36.8347|[-86.8185, 174.756]|False|True|True|True|
|forward_08|104.068|[65.2879, 146.733]|False|False|False|True|

No-policy strict-USD standing20s: mean absolute drift 0.00099725708deg; max 0.0016665662deg; falls 0/1024.

## Full cell metrics and paired S/D

All individual training-seed values, group means and95 intervals are in [seed_tables.md](seed_tables.md), [cell_seed_statistics.csv](cell_seed_statistics.csv) and [paired_seed_statistics.csv](paired_seed_statistics.csv). Per-trial paired S/D, common-prefix exposure and half-swap residuals are losslessly retained in paired_raw/. Cell/job statistics bind the original raw array SHA256 values.

## Videos

Selected USD symmetric, training seed 5, because it has the largest absolute seed-mean D (+7.03253deg). Evaluation seed1/env0 is fixed for both phases; it is not the maximum-D trial. Exact formal saved states, no re-integration and no additional evaluation. Both phases have colored front/side/overhead views, plus synchronized side-by-side overhead; common worldXY scale and heading arrows. Front/side cameras follow translation and heading; overhead follows translation with fixed world orientation. Original HumanML3D NPY: N/A, stage1 velocity command rather than motion retargeting.

[k0 front](videos/phase0_front.mp4); [k0 side](videos/phase0_side.mp4); [k0 overhead](videos/phase0_overhead.mp4); [k10 front](videos/phase10_front.mp4); [k10 side](videos/phase10_side.mp4); [k10 overhead](videos/phase10_overhead.mp4).

[Paired overhead](videos/phase0_phase10_overhead_paired.mp4); [actual visual review](manual_visual_review.json); [joint colors](joint_colors.json).

Four plots: [paired_S_D](figures/paired_S_D.png), [heading_startup_and_slope](figures/heading_startup_and_slope.png), [halfcycle_swap_residuals](figures/halfcycle_swap_residuals.png), [falls_and_exposure](figures/falls_and_exposure.png).

Actual review covered four plots, two contact sheets, eighteen full-resolution single-view frames and three paired frames. Colors and complete connected body verified. Arms remain commonly outstretched; no natural-gait claim. Display velocities are zero; saved poses replay without physics integration.

Transparent corrections: [prefix precision correction](post_evaluation_analysis_correction.json) fixes only derived common-prefix float64 reductions; all 1110 raw inputs and primary gate metrics unchanged. [Color correction](media_color_correction.json) rejects gray drafts and uses stronger link-root visual material binding; physical attributes unchanged.


## Budget, storage and historical status

No training, parameter scan, extra candidates, default promotion or reward change. Required gates and media are distinct from formal denominators. New work stays on the remote server; no mounted files copied to the local computer. Per prior user authorization12 unreferenced obsolete SuperMimic intermediate weights were removed only after retaining and hashing their final siblings; receipt and per-file hashes are archived. Issue17 scientific acceptance remains failed; SuperMimic remains stopped. GitHub issue and Notion receive the same findings; Linear is not used.
