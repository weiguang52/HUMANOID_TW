# Issue11 prospective analysis plan

Base5fa882398b4edfb7ac0a93dc5607822c62ae8a43. Issue11: analysis only,
no training, no policy/reward/randomization/default changes. Fixed requested
training seeds1/2/3/4/5, checkpoints1500 and4500; all retained. #7 protocol:
bounded combined,1024env,eval seeds1/2/3,forward0.3/0.5m/s,20s,deterministic,
paired reset. Thirty eval jobs,61440 trials. Same command indices0/1 as prior.

Opt-in observational logger reuses FailureRecorder and original push sampler
exactly once, no random draws/physics writes. Before formal data, run64env
baseline-recording versus expanded-recording smoke and compare final states,
RNG states, first-failure outcomes/traces. Formal counts compared with#10.

Record all trial initial masses/nominal ratios for every link, all shape
static/dynamic friction/restitution, body-to-shape mapping, initial delay.
Record EVERY push including post-first-failure ones, with first_trial_at_risk
flag, world/body actual delta-v, step/time, support feet/contact force,
contact-onset inferred cycle phase, root/whole-bodyCOM state and foot positions.
Risk analyses use only pre-first-failure exposure. Post-reset pushes are archived
but excluded. First failure pre-reset traces and terminal reason remain intact.

Trial unit: one initial environment/command over20s, first true termination.
Baseline mass/friction/delay risk tables include EVERY successful/failed trial.
Foot-mass constant ratios are reported as degenerate/not estimable. Primary
friction scalar is minimum assigned static coefficient over both foot shapes;
dynamic minimum is a separate secondary factor, not silently substituted.
Use common outcome-blind initial-friction quartiles across both checkpoints,
deduplicated by eval seed/env when values identical. Base mass cuts=.75,1,1.25
from configuredU(.5,1.5). Push3D norm preserves world delta norm; its configured
square-uniform quartiles are .3*sqrt(1/pi),.3*sqrt(2/pi),.3*sqrt(3/pi).
Directions: forward/back if|dx|>=|dy|, otherwise side (record left/right too).
Support codes0flight,1left-only,2right-only,3double. Supporting-foot cyclephase
since same-foot onset/previous period, -1 unavailable; phase quartiles0/.25/.5/
.75/1 plus unknown. These are inferred contact phases, not commanded gait phase.

Push unit: an at-risk push followed for1s, only if the administrative20s horizon
allows full followup. Failure within(0,1s] is an event. If another push occurs
before1s, censor at that push and report excluded counts; failure before it
still counts. Intervals ordinarily5-8s, but check actual overlaps. Never label
post-failure pushes as independent successful trials. Report fixed-baseline
trial risks and push-window risks separately. Wilson intervals are descriptive
Bernoulli intervals; repeated pushes/trials and shared physics draws violate
simple independence. Main-factor effects also report equal-weight seed risks,
paired seed-bootstrap contrasts (5^5=3125 indices), smallN caveats. Effects are
observational associations, not adjusted causal effects; joint stratified tables
show confounding/sparsity explicitly. No outcome-based quartile threshold tuning.

Predefined corners: STRICT=staticQ1 & baseQ4 &20ms & bodyforward delta>=.3m/s
& single support. ENVELOPE=(staticQ1 OR dynamicQ1) & base(Q1 OR Q4) &20ms &
ampQ4 & single support. Both fixed, report zero/sparse cells honestly. Primary
concentration diagnostic uses ENVELOPE: rare if<=20% trials have an eligible
at-risk corner push, dominant if>=80% first failures occur within1s after one.
Only if both conditions hold call few measured corners dominant; otherwise
report broader failures relative to these predefined corners, not proof that
all conceivable corners are excluded. Report per-seed sensitivity and bootstrap.
For both corners report trial coverage, all-failure co-occurrence, proximal
failure fraction, push risk and complement risk. Exclude WHOLE trials with any
eligible pre-failure corner exposure for diagnostic remaining20s fall rate;
never remove only failed trials. Exposure membership depends on survival time:
this is descriptive selection, not a counterfactual intervention or acceptance.

1500vs4500 uses identical cuts/denominators/protocol. Show changes in each
baseline and push stratum, common-support counts and first-failure mode shares.
Do not infer physically unrecoverable combinations from this policy failing.
COM/foot support and friction/capture-point-style diagnostics may indicate stress,
but actuator/contact/control limits and alternative policies are untested.
Recommend exactly ONE next direction conditional on concentration; do not implement.

Representative videos: two highest-count4500 low-speed failure strata among
20ms/0ms x last-push direction x support, include no-recent-push as case label.
Risk tables never use case-label counts as trial denominators. For a push stratum
require>=20 eligible pushes; choose failing case nearest median push magnitude,
ties seed/eval/env/step ascending. If no recent push choose earliest failure.
Replay EXACT recorded root/joint pre-reset states, not fresh1env re-simulation.
Colored pre-baked exact USD, preserve original physics/non-material attributes;
front+side, follow translation/heading, joint legend, video/frame/color/fullbody/
attachment QA. Stage1 velocity commands have no HumanML3D sourceNPY(N/A).
All files remain remote; Notion replaces Linear. Commit/push results and verify.
