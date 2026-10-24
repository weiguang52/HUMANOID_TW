# Standing heading pilot, 2026-10-24

Continuation of TW-RL Notion004. Existing support velocity candidate is rejected.
Goal: test one low-weight stationary heading term without changing observations,
physics, sampling, contact reward, position reward, or formal checkpoint.

Audit: reference and actual velocities are world-frame body velocities in commands.py.
Contact sidecar validates SHA256, clip order, cropped lengths and foot order. The
stationary support gates depend on reference labels, not actual support; both
static diagnostics have double-known support fraction1.0. No frame mismatch found
in code review; this is not proof of every retargeted derivative being perfect.
The candidate upper diagnostic reaches25.91deg heading error at5s vs12.36deg
baseline. Static drift grows263.91mm at15s while heading error is only-6.03deg:
heading cannot explain all translation drift. Measured support speed rises from
0.0140 to0.0204m/s. Detailed evidence is prior_diagnostic_audit.json in state.

Train manifest:1016standing clips/634807frames and332walking/153960frames.
Sampling is failure-aware with an explicit prior; frame ratio is not actual
sampling frequency. Do not infer category starvation from dataset counts alone.
No sampling changes in this pilot.

Implementation: HeadLocked plus stationary_heading weight-0.05, angular scale
0.35rad, smooth reference speed/yaw gates matching the audited standing definition.
Wrapped yaw Huber cost avoids the old velocity exponential cost saturation.
No contact gate on heading: pose tracking must remain active through stance shifts.
This is a hypothesis test, not a demonstrated improvement.

Validation:27 pure tests pass, including angle wrapping, quaternion sign, common
yaw rotation invariance, moving/turning/category exclusions and finite gradient.
Both64env2iteration smoke runs completed. Formal matched pilot:1024env,seed42,
500iterations from the same model_27995.pt; save every100iterations. Predetermined
final checkpoint evaluation, no best-of-test-set selection. Independent selection
set required before tuning intermediate checkpoints in any subsequent experiment.

Acceptance: static drift less than original109.15mm and matched baseline; upper
diagnostic completes20s; standing6/6, flat at least23/24, turning3/3 retained.
42cases include diagnostic categories; aggregate success cannot override regressions.
Report colored front/side and original HumanML3D videos, including early termination.
Do not promote without verified metrics and visual review.

State:/root/gpufree-data/datasets/practice9/heading_ablation_20261024
Scripts:train_heading_ablation.sh,prepare_heading_evaluation.py,run_heading_experiment.sh.
Status:pilot launching; optimization outcome pending.

## Result: 2026-10-24 23:39 Asia/Shanghai
Both pilots and assemblies completed; model_28494.pt exists for each.
Each delivery contains46videos,28dual-view encoding checks and14front-camera audits.
Baseline34/42; heading35/42. Flat24/24 both, standing5/6 both,
turning2/3 vs3/3, reference-limit3/3 both, treadmill0/3 and nonplanar0/3 both.
Static20s drift227.95mm baseline vs261.12mm heading (original109.15mm).
Upper terminates7.52s baseline (anchor orientation),6.52s heading (anchor position).
These are not completed20s trajectories. Standing heading RMSE13.73deg vs21.65deg;
wrist7.29mm vs7.46mm. Reject both for promotion; original checkpoint unchanged.
Baseline also regresses: do not attribute all regression to the added term.
One training seed and500iterations limit conclusions. Remaining optimization
requires investigation of continuation stability and contact drift.
