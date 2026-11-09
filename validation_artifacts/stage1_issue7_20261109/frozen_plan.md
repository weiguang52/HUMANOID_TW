# Issue #7 frozen high-resolution failure-analysis plan

Base commit36e38eed1c2c93b87248582af6083b2df83b3065.
Issue https://github.com/weiguang52/HUMANOID_TW/issues/7 explicitly says analysis only: no retraining/default changes. The user repeated a general training request; clarification was requested while independent analysis proceeded. Pending contrary direction, follow the specific issue scope.

## Fixed protocol
Four policies: issue6 bounded_strengthen training seeds1/2; issue5 strengthen seeds1/2. Use the same bounded combined evaluation for all, as the matched comparison in issue6. Original evaluation push +/-0.3m/s (not training +/-0.45), mass nominal*U(0.5,1.5), unchanged friction, noise and random0/20ms delay.
Each policy uses1024 environments, eval seeds1/2/3, commands0.3/0.5/0.8m/s,20s per command. Paired reset seed=eval_seed*1000+command_index; deterministic policy. Twelve jobs,36 command-level estimates,36864 trials. Report per-seed Wilson95% binomial intervals and pooled3072-env conditional intervals separately; do not treat environment CIs as training-seed uncertainty or proof of <1%.
Recorded/plain64-env seed77 smoke yields12 falls in both and bit-identical metrics; actual push deltas validated within original range. Wilson implementation checked against SciPy across boundaries/interior.

## Recording
First true termination per environment only, up to1s (51 frames at50Hz) ending in terminal pre-reset state. Termination-manager hook captures after physics/termination computation and before automatic reset. Push wrapper calls original sampler exactly once and records actual velocity delta without extra draws. Track root local pose/quaternion/velocity, all joint positions, base height/roll/pitch, left/right foot forces/contact and cycle phase, current delay, latest world/body-frame push, elapsed push time and contact at push, command, hard/soft joint-limit proximity flags.
Cycle phase: time since same-foot contact onset divided by previous onset-to-onset period, clamped0..1; -1 unavailable. Contact phase: flight,left-only,right-only,double support. Joint proximity threshold0.02rad; this is not proof of physical stop impact.
Record per-delay trial denominators plus at-risk step exposure for delay/contact/limit/recent-push groups. Tables of case proportions are not causal risk estimates. Post-failure auto-resets do not create new independent trials.

## Predeclared descriptive categories
Recent push: recorded latest push age in[0,1]s; older/no push distinguished in tables. Push body-frame direction grouped by dominant x/y component and sign; magnitude bins[0,.15),[.15,.30),[.30,infinity).
Terminal tilt: lateral if abs(roll)>=abs(pitch), otherwise sagittal; near-upright when both<5degrees. Primary mode is recent-push/not-recent crossed with terminal tilt. A mode is called dominant only if>=50% of all first failures and ranked first for each of4policies; otherwise state no dominant mode. Report per-policy/speed distributions to expose heterogeneity.
Select most frequent1-2modes for evidence plots and colored front+side replay. Representative has a full1s window and is nearest median failure time within mode, deterministic tie by source path/env id; selection is diagnostic, not performance evidence.
Recorded-state kinematic replay of exact root/joint trajectory at native50Hz, no policy rerun or physics integration; label it clearly. Exact colored asset preserves physics attributes, camera follows translation+heading. Velocity commands have no HumanML3D source NPY.
Propose one next modification based on evidence, but do not implement it. Preserve negative findings/limits. All artifacts remain remote; update Notion and issue; commit/push and verify.
