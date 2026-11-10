# Issue13 frozen diagnostic protocol
Base: 0ec49f358d1634b62b7f8b149381809b54d23b9b.
Issue13 created2026-11-10T15:49:04Z: analysis only; no training or default/configuration promotion.
Ten exact issue12 final checkpoints, five control/treatment training-seed pairs. No model/std/optimizer updates. All source training/default configuration and original evaluator remain byte-identical to base.

## Formal cohort
Clean: ten models x evaluation seeds1/2/3 x1024envs x20s xfive separate commands: forward03(.3,0,0),forward05(.5,0,0),forward08(.8,0,0),yaw_positive(0,0,.3),yaw_negative(0,0,-.3).
Clean uses existing clean evaluation switch: no push/delay/observation noise/mass/material randomization; actual nominal per-link masses and authored material coefficients must be audited, along with terrain nominal friction and ordinary JointPositionAction class.
Combined: same ten models xthree eval seeds x1024envs x20s xforward03/05/08; exact issue12 bounded-combined configuration, original command order and seed*1000+command_index resets. Re-run is necessary because old full-trial files contain pushes/first failures but not full-horizon velocity moments.
Total60formal jobs,240command records,245760initial trials(153600clean+92160combined).
Clean retains the ORIGINAL evaluator order: forward03,yaw_positive,yaw_negative,forward05,forward08; combined preserves issue12 order. This choice was made before any smoke or formal data, to permit direct original-evaluator invariance validation. Deterministic policy; paired resets within protocol and across models. No claim of identical physical distributions across clean/combined.
Combined all-trial/failure recorder is unchanged; assert raw arrays and original summary metrics reproduce issue12 for every job.

## Time/sample definitions
Primary metrics retain issue7/12 semantics: non-terminal post-step samples across the full20s, including ordinary episodes after a failure reset. Each environment still has a full-horizon trial identifier. Report a separate initial-episode-only sensitivity analysis (stop each environment at its first reset), without replacing the primary definition.
Primary recovery partition is post-event time [push,push+1s): post-step sample at push step k is inside, k..k+49 atdt=.02; outside is its complement. Observer clocks clear on any reset BEFORE interval push events, so no window leaks across episodes. All initial/surviving trials remain represented; segment sample counts/durations are reported.
Also retain reward-aligned boundary sensitivity k+1..k+50 (original training reward was computed before interval pushes). Do not silently equate these two post-step sampling conventions.
For each trial/scope/partition save count, float64 sum and sum of squares for xy norm error,yaw absolute error,vx,vy,wz. Save per-step cohort moments, packed sample/window masks, push timing, and raw sampled reference-env values for independent reconstruction audits.
Bias = mean body-frame vx minus target vx; fluctuation = temporal population SD of body-frame vx around its own trial mean. Also report pooled temporal/environment variability separately from the mean of per-trial temporal SDs; do not conflate between-environment offsets with temporal fluctuations.

## Seed-level inference and frozen A/B decision
For each model/command/segment, primary error is pooled valid-frame error over1024envs, matching issue12. Average the three evaluation seeds within each training seed, then weight five training seeds equally.
Use all5^5=3125 paired joint training-seed bootstrap index resamples, percentile95%, no multiple-comparison correction. Give paired absolute and relative group-mean changes and intervals, not environment-level inferential pseudo-replication.
A: ALL clean xy/yaw metrics for ALL five commands, and combined OUTSIDE-window xy for ALL three speeds, have treatment group mean <=1.05*control mean.
B: ANY of those same13 comparisons has treatment mean >1.05*control mean AND the paired absolute-difference95% lower endpoint >0.
Otherwise: unable_to_distinguish. A requires the point bounds, not a newly invented confidence non-inferiority rule.
Handle zero control means explicitly: T=0 passes, T>0 is a degradation; never invent infinite ratios or pseudocounts.
Only one next-step recommendation, chosen after this frozen rule: A asks user to decide whether future acceptance should evaluate clean/outside tracking; B proposes one structural modification with evidence, no implementation; otherwise recommend retaining current decision until uncertainty is resolved. Do not change issue12 rejection or any default.

## Observer validation before formal jobs
Unit checks for window boundaries/reset, sufficient-moment/variance identities, known paired-bootstrap/gate cases.
Combined and clean smoke: original versus observational adapter, same64envs/seed/checkpoint/command order; compare every original metric, final root/joint/complete CPU-CUDA-NumPy-Python RNG, and original combined all-trial arrays bitwise. Original source and model checkpoints SHA locked. No action/event/observation/reward edits.
No formal outcomes used to choose gates. Freeze implementation hashes after smoke, before all formal jobs.

## Representative exact-state videos
For each group choose the combined0.8m/s trial with valid-frame xy error nearest that group's median across15360trials; deterministic ties: training seed,eval seed,env id. No survival/performance filtering. Selection is an illustrative median, not a new acceptance sample.
Reconstruct the selected two original1024env evaluations only to capture exact full-horizon selected-env states; require all original metrics, push records and sufficient moments to match formal runs. These two reconstruction jobs do not enlarge formal denominators.
Replay exact recorded states without physics re-integration: four20s colored front/side clips with shared sample times per selected trial, push-time table and on-video push timeline/1s window labels. State whether any reset occurred. Camera follows translation AND heading. Preserve precolored exact experiment physics, legend,1000frames/50fps; inspect colors/full body/attachments/encoding and sample frames.
Stage1 velocity commands have no HumanML sourceNPY: explicit N/A.
All data stays on remote disk; archive analysis, full per-trial sufficient records, model references, diagnostics and videos; commit/push remotely, verify SHA/tree, reply issue13 and sync Notion Done.
