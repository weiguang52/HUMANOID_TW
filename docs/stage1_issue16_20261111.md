# Issue #16: frozen heading gate and source-discrimination protocol

Base commit: 770014ee9a4393da6dff34eced89a04d83e788dd.
Issue: https://github.com/weiguang52/HUMANOID_TW/issues/16 .
Client date: 2026-11-11 (Asia/Shanghai).

## Scope frozen before new simulation

Evaluation and provenance only: zero training updates; no reward, robot, actuator, startup/reset, or production observation/configuration changes. No default promotion. The only intervention is an inference-side sagittal reflection wrapper; a no-policy test uses default joint-position targets. All original production files and all 35 original checkpoint bytes are preserved. No mounted data are copied locally. Existing issue15 raw data provide the entire lineage gate table without re-evaluation.

## Heading acceptance gate

For stand and forward0.3/0.5/0.8m/s (yaw target0), define initial-trial cumulative heading H=sum wrapped successive wxyz-quaternion Euler-yaw increments over nonterminal steps, excluding reset jumps, in degrees. A complete20s endpoint exists only for a no-reset initial trial; failed trials keep initial-prefix signed/absolute heading, prefix duration and reset-excluded all-episode-segment heading, never fabricated20s endpoints. Report originalN, failures and survivor counts.

Within each training seed average the three evaluation-seed means, then average training seeds equally. The numeric heading gate is mean absolute20s survivor drift <=30 degrees for EACH of the four commands. Also report exact training-seed percentile bootstrap95 upper bound (all n^n draws for n=2/5), signed means and seed-level values. The issue asks to report the upper bound, not to silently replace its mean<=30 gate by upper95<=30: both mean-gate and conservative-upper-bound flags are shown separately. A cell with missing endpoint statistics in any evaluation/training seed is not a demonstrated pass; label incomplete/censored and show failure burden. Single-training-seed nodes have no training CI. All-four node passes require all four finite mean gates; no hidden fall/xy relaxation is introduced.

## New formal evaluation budget

Mirror groups: control6000 training seeds1..5, requested4500 seeds1..5, and the one available robust05 parent. Three evaluation seeds1/2/3 per checkpoint,1024env,20s per command. Evaluate the six original issue15 commands in their unchanged order (forward03,yaw+,yaw-,forward05,forward08,stand), preserving original paired START reset ordinals0..5; yaw+/- are supplementary controls. Thus33 mirrored jobs,198 command records,202752 initial trials. Original counterparts are immutable issue15 data, not new baseline evaluations.
One no-policy clean job:1024env, evaluation seed1,stand20s, default joint-position target every control step, same original termination/reset protocol; no artificial balance controller. If it falls, report prefix/selection limitations rather than claiming full20s standing.
Total new formal jobs34;203776 initial trials. Zero-update mapping/observer smoke jobs and one exact source-state reconstruction for the representative video are separate validation, not independent formal replications. No checkpoint search, extra training or budget expansion.

## Mirror definition and preflight

Reflect in sagittal plane S=diag(1,-1,1). Polar vectors map S*v; axial angular vectors map -S*w. Commands(vx,vy,wz) map(+,-,-). Left/right joints are exchanged, with signs derived and checked from actual joint axes/frames, including central waist/neck joints. Mirror physical joint positions/velocities and desired action targets; relative observations/action offsets and per-joint scales are accounted for explicitly. The global gait sin/cos pair is unchanged: a pure spatial reflection does not change elapsed time. The evaluation conjugates the original policy at the SAME clock, rather than combining reflection with a temporal phase intervention. Apply reflection to raw101-wide actor observations BEFORE existing policy normalization; reflect returned actions back. Critic input is not used to drive inference and is not changed.

Before formal runs publish the per-joint permutation/sign/offset table, actual observation term/order/scale and action order/scale/defaults. Tests cover two mirrors=involution for arbitrary and symmetric states, polar/axial signs, global-clock invariance and clock-driven spatial policy conjugacy, desired-target equivalence, joint pairing/limits and normalization order. Validate runtime order and physical coordinates against actual USD/URDF, not joint names alone. Identity-wrapper smoke must reproduce original policy, physical state and all RNG states bitwise. Mapping tests passing alone do not establish simulator symmetry.

## Interpretation frozen before new outcomes

At each zero-yaw command report original/mirrored signed and absolute heading, per-seed sign reversal, difference and signed-sum intervals, failures and valid exposure. Supplementary operational magnitude comparability is mirrored/original mean absolute drift in[0.8,1.2], with raw values and training-seed intervals; near-zero signed values (|mean|<=1degree) are direction-indeterminate. This descriptive band is not a trained-policy acceptance threshold.

Classify policy bias / robot-simulator asymmetry / both / unable to distinguish by converging mirror response and physical audit/no-policy evidence. Reversal of inherited policy bias does not prove a perfectly symmetric robot; same-direction drift does not uniquely localize to USD if simulation, gait timing, reset asymmetry, censoring or nonlinear interactions remain. Verified robot asymmetries are reported even if their causal contribution to drift remains unproven. Mixed/unstable responses or confounded failures are explicitly unable-to-distinguish where appropriate; do not force a binary causal claim.

## USD and runtime audit

Audit the exact experiment USD urdf0711_training_30dof_tw123_yaw_waist_v3.usd and resolved referenced assets, recordingSHA. For every left/right pair compare mass, center of mass, full inertia tensors in a common reflected frame, joint local frames/axes/limits, default angles, PD gains/armature/effort/velocity limits, and actual collision meshes/primitives including points/transforms/topology/convex cooking settings. Include unmatched central links and signed central limits. Separate exact structural differences, numeric tolerances and runtime readback. No model correction is implemented.

## Media and publication

Select control6000 forward0.5 trial nearest the pooled initial-trial median absolute20s heading drift from all15360 initial trials, with failures/selection explicitly disclosed (no silent survivor filter). Reconstruct exact original evaluation state and verify all original records/arrays. Deliver colored front/side AND overhead-following evidence if needed, world-XY trail plus reset-aware numeric heading. Saved precolored USD must preserve exact experiment physical attributes; no dynamic binding. HumanML source NPY is N/A for Stage1 velocity commands. Actually inspect colors/body/limb attachment/heading and figures before claiming manual QA. Archive all new raw data/analysis/tests/configuration provenance remotely; only one next recommendation, not implemented. Commit/push exact files, verify actual GitHub branch/tree/parent and issue reply, update Notion (never Linear).

## Preflight operator-definition correction (before formal runs)

The initial draft negated both global gait-phase features. Independent coordinate reasoning showed that this would combine spatial reflection with a half-cycle temporal intervention. The physical mirrored trajectory must retain the same exogenous clock, so the final operator leaves sin/cos unchanged. The initial64-env runtime preflight was stopped, preserved separately and excluded from scientific analysis. No heading result or acceptance threshold was used to choose this correction; no formal job had started. A clock-driven conjugacy counterexample test now detects the invalid phase shift. The30degree criterion,33 mirrored jobs plus one no-policy job, checkpoints and all other budgets are unchanged.
