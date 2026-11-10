# Issue12: one post-push pitch recovery penalty, five matched training seeds

Issue: https://github.com/weiguang52/HUMANOID_TW/issues/12
Base branch/commit: supermimic/tw-locomotion-20261108 / 67d9879.
No production-default promotion; the new reward is opt-in and otherwise absent.

## Frozen scope and calibration rule (before formal training)
Exactly one penalty: -w*(pitch^2 + 0.1*pitch_rate^2), during the 1.0s after every
original push, zero otherwise. Pitch is ZYX Euler pitch from root quaternion;
pitch_rate is its exact derivative cos(roll)*body_omega_y - sin(roll)*body_omega_z.
No direction, speed or contact gate is added. Timing is privileged reward-only
state and never enters actor/critic observations. Integer global control-step
timestamps survive episode-length randomization; reset clears only reset IDs.
The installed env computes rewards before reset and interval pushes. Consequently
an actual push at counter k affects the next 50 rewards at k+1..k+50 (dt=.02).
The instantaneous cost helper is active also at age0; there is no second reward
computation at the push instant. A new push restarts only that environment window.

The unchanged original push sampler runs exactly once. Both experiment arms use
identical pure timing/reset instrumentation; only the new reward weight differs.
Without the explicit CLI weight, original configurations/events are unchanged.

Calibration: five fixed 4500 parents, original robust08 strengthened-push bounded
training distribution,4096envs,1024control steps each, seeds1..5, random initial
episode lengths as in training. Frozen loaded actor samples stochastic actions;
no optimizer, parameter or normalizer updates. All steps, including normal reset
cycles, count. Compute equal-seed R=mean(old total reward per control step)>0 and
Q=mean(gated pitch^2+.1*pitch_rate^2)>0. Fix w=.05*R/(.02*Q), with no rounding or
weight/performance scan. Save all inputs and one frozen_weight.json + SHA before
formal training. Invalid/nonpositive calibration is an error, not a weight search.

## Invariants and fixed budget
Nine timer/config unit tests; runtime original-vs-zero-vs-treated smoke checks
for complete RNG, root/joint, rewards and actual observation dimensions. Actor,
critic and optimizer load must match each parent's state exactly, including std.
Five pairs: each own 4500 parent -> control0 / treatmentw,4096envs,+1500updates.
Run order control_s1,treatment_s1,...,control_s5,treatment_s5. No std reset, no
checkpoint choice, no extra budget. Parent file model_11994.pt represents4500
actual cumulative updates; final labels use logged new updates, not file names.
Per-iteration monitoring includes dt-weighted new/old/total rewards, active-window
fraction and xy/yaw tracking rewards. Report drift from the initial5% calibration;
never tune w during training. Freeze source/config/checkpoint hashes before runs.

## Evaluation and decision
Final10models x eval seeds1/2/3 x commands0.3/0.5/0.8m/s x1024initial trials,
20s deterministic, paired resets, bounded combined #7 protocol:92160trials.
Use #11 full-trial/push recorder including survivors, original frozen amplitude
quartile cuts, all denominators and administrative/post-failure exclusions.
No treatment reward or privileged timing is needed for policy evaluation.

Unit of inference: five paired training seeds, conditional on shared upstream
parent and fixed evaluation streams. Uniform paired seed bootstrap over all
5^5=3125resamples; percentile95% intervals, no evaluation-env pseudoreplication.
Primary effect = treatment minus control0.3m/s fall rate: effective only if the
entire95% interval is negative. Treatment0.3 mean CI upper must be<1%; treatment
0.5 mean must be<=control. Tracking: at each of0.3/0.5/0.8m/s, treatment mean xy
error must be<=1.10*control; yaw is reported. Metrics retain #7 definitions:
non-terminal frame means, including later normal auto-reset episodes. State this
scope rather than implying first-trial-only velocity samples. Zero reference
tracking is handled explicitly. Secondary: Q4-forward1s failures and delay20/0RR
under the same #11 cuts, with denominators and uncertainty. Report all seed pairs,
means, sampleSD, failure counts and limitations. An ineffective/slow scheme stops;
no more budget or weight scan. Production remains opt-in/off.

## Delivery
For each of10final models choose the highest pooled three-eval-seed fall-rate
command; ties prefer0.8,0.5,0.3. Supply colored front+side20s illustrative one-env
rollouts, fixed seed1. These videos are not the1024-env acceptance sample or an
exact original failure. Use exact precolored experiment USD, preserve physics,
locked joints and non-material attributes. Camera follows translation/heading;
legend, encoding/frame/framing/color/attachment audit. Stage1 velocity commands
have no HumanML3D source NPY (N/A). All files stay remote; Notion replaces Linear.
Archive all final models, metrics, config/load/source audits and analysis; commit,
push and verify remote branch SHA and file manifests; synchronize issue/Notion.
