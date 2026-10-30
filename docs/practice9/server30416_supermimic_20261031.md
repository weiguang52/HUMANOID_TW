# Server30416: migration, SuperMimic adapter and final trial

Status,2026-10-31: environment migration, original Isaac training/recovery,
SuperMimic implementation, actual training/recovery and fixed-test video
verification are complete. **The1000-iteration learned policy failed all seven
holdout motions and is rejected for production use.** No existing policy was
replaced. This concludes this bounded migration/method-transfer trial, not the
broader TW-RL stability work. Subsequent research remains open.

## Storage and runtime

Target root@183.147.142.40:30416; repository
`/root/gpufree-data/projects/HUMANOID_TW`, branch
`codex/support-velocity-20261024`. Latest verified pushed revision is recorded in
`/root/gpufree-data/migration/latest_push_receipt.json`.

All bulk files, environments, generated assets, datasets, models, temporary
files and caches remain under `/root/gpufree-data`. Source→target transfers were
direct, without local dataset/media copies. Existing `/opt/conda` is a bootstrap
read dependency; drivers remain system-provided. Target GPU is RTX4090/24GB.

- Original runtime: source `scripts/practice9/activate_data_environment.sh`;
  environment `conda_envs/env_isaaclab`, copied Isaac Sim/Lab retained.
- Independent SuperMimic runtime: source
  `scripts/practice9/activate_supermimic_environment.sh`; environment
  `conda_envs/env_supermimic`, Python3.11/Torch2.7+cu128/mjlab1.6/MuJoCo3.11/
  Warp1.18. Adapted fork installed editable; isolated `python -I` resolves it.
  `pip check` and actual GPU/lazy-linalg computation passed. Activation scopes
  CUDA/Torch/library loading and caches to this data-disk runtime.

Migration evidence under data `migration/` includes original manifests,
`combined_content_audit.json`, failed-entry rechecks and explicit link relocation
receipts. Coverage:509267 main entries,14376 legacy-run entries,6434 validation
artifact entries;43 mesh and198 importer files were separately checked. Failed
initial audits remain preserved; combined exact rechecks passed. Files requiring
relocation were resolved on the data disk; no system-disk bulk installation.

Original Isaac64-env smoke completed2 PPO iterations/48 control windows and a
new-process1-iteration reload/24 windows, with finite changed actor/critic,
195→203 observation expansion and Adam progression560040→560060. Evidence:
`datasets/practice9/migration_smoke_20261031/{smoke,reload}_migration_audit.json`.
These prove execution/recovery, not learned skill or convergence.

## Method provenance and TW adapter

[Paper](https://arxiv.org/abs/2610.28480),
[author project](https://supermimic.github.io/).
No official author implementation was linked when checked. The adapted
Apache-2.0 third-party implementation is
[YHFone/supermimic-mjlab](https://github.com/YHFone/supermimic-mjlab), pinned to
`9aab0a588c6941bf9862a2d444b711fd73e2b832`. Original36 files were Git-blob
verified. Preserve `third_party/supermimic_mjlab/LICENSE`, `TW_UPSTREAM.md` and
source audit. This is a third-party method transfer, not official reproduction.

The implementation actually alternates simulator collection, multi-step world
model fitting, actor BPTT through a temporarily frozen world model and critic
bootstrap. World model training continues each iteration. Full upstream network
widths remain; the adapter is not merely a PPO reward change.

The robot uses30 movable joints,14 tracking bodies and chest anchor. PD gains,
effort/armature/action scales, nominal shoulder-roll readings(-/+pi/2), and
joint/action mapping match the original TW task by names. Zero normalized action
means nominal-pose PD targets, not zero torque or reference feedforward. All
head joints remain controllable; the previous head-locked task is not imposed.

Source URDF is unchanged:
`datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf`, SHA256
`1d613d3aef37d99e467ac7574d1455dacd3e9c5cc3276feff06afda6b25386f2`.
Mass0.840742808746342kg, nq37/nv36. Exact STL→OBJ conversion retains every
oriented float32 triangle; no mesh decimation or inertia balancing. Restored
17-digit MJCF numbers pass independent URDF/MuJoCo comparisons over50 poses/all35
links: maximum position1.665e-16m, rotation matrix9.992e-16, inertia6.242e-12,
COM0. All332 training clips/153960frames pass14-body FK with position1.076e-7m
and orientation9.215e-7rad maximum errors. Final holdout FK also passes.

Assets and audits are in data `supermimic_tw_20261031/robot/` and
`projects/supermimic_research/`. `tw30_colored.xml` transfers colors offline from
the precolored USD;66 geoms,30-joint palette,481 non-color compiled arrays
unchanged. MuJoCo does not directly load USD. Same robot geometry/inertia are
used for training, policy evaluation and reference rendering.

## Correctness fixes and invalidated trials

All pre-physics-v3 SuperMimic runs are invalid for acceptance and retained only
for diagnosis. Do not reuse their weights: earlier default-pose matching and
independent velocity clipping learned a different/invalid plant. The earlier
1700-iteration continuation plan was canceled. The obsolete untracked queue
script was archived to data research with a SHA256 receipt.

- mjlab first-match patterns let `.*:0` shadow shoulder defaults. Explicit named
  defaults now match the original task, with1.19e-7Nm nominal PD residual.
- Independent hinge speed clipping injected floating-base momentum. The
  replacement uses an FP64 mass-metric box QP and free-root reaction before
  native position integration. It checks finite state, convergence, free
  generalized momentum and non-increasing kinetic energy. Native Warp `_advance`
  code retains its Apache attribution/source fingerprint. Eager shared-stream
  Warp/Torch stepping is used. This operator-split speed/contact method is not
  claimed exactly equivalent to PhysX.
- Encoder context is mean friction+J biases+3 COM values; the upstream offset
  mixed the last bias with COM. Shared named-layout extraction fixes consumers.
- Replay uses actual complete within-episode windows. No retry fallback across
  resets, padding or shortened dynamics windows.
- Imagination history is strictly past observations/actions before appending the
  current frame; reset padding and oldest-ring previous actions are handled
  explicitly. No duplicate-current/future-action leak.

Physical contract is `explicit_default_pd_mass_metric_velocity_fp64_v3`;
context/replay/history tags and state/action orders are checked on resume.
Leg/foot speed caps are3rad/s; waist/arms/head2rad/s. Native inertia decoding
matches CPU full inertia within1.77e-8. Static zero-policy nominal calibration
runs20s/1000 controls without termination: max horizontal2.880mm, height2.149mm,
orientation0.01166rad. This synthetic test is not learned standing success.
Forced +/-10rad/s initial-speed test exercises the projection: residual4.50e-7,
free momentum error2.50e-8, kinetic-energy increase0, with TF32 enabled elsewhere.

44 fast tests passed. Actual corrected TW-reward4-iteration training and
4→6-iteration new-process recovery passed finite changing WM/encoder/decoder/
critic and Adam progression(8→12,4→6,16→24). Checkpoints save all three optimizer
states, RNG and fractional collection accumulator. Replay, simulator and sampler
statistics rebuild on resume; continuation is not bit-identical simulator replay.
Evidence: `physics_v3_{smoke,reload}_audit.json`, `physics_v3_runtime_audit.json`,
`static_mass_metric/report.json`, `impulse_mass_metric_fp64/report.json`, plus
JUnit `projects/supermimic_research/physics_v3_tests.xml`.

## Actual fixed1000-iteration trial

Data state root: `/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/`.
Final run:
`/root/gpufree-data/twrl_runs/supermimic_tw_20261031/physics_v3_1000/2026-10-31_05-15-56`.
Final `checkpoint_final.pt` SHA256:
`507a1b865cb843a5a273520e022e485ab039b2a44ee96c79582021872e373340`.

Fresh initialization,seed42,256envs,1000iterations,H32/window8,WM batch512×10
updates,policy batch512/chunk128,buffer1000,2.5 collection controls/env/iteration,
checkpoint every100. Recipe is `scripts/practice9/run_tw_supermimic_physics_v3.sh`;
existing-output guards intentionally prevent relaunching this completed trial.
TW reward widths reuse original core scales(position.15m, body/root orientation
.5/.4rad, linear/angular velocity.75m/s and2rad/s), keeping SuperMimic form and
weights. No failed experimental contact rewards were imported.

Training used `training_speed_blend_20261026/train.json`:332 clips,331 source
families,153960frames at50Hz(~51.32min reference), all labelled `walking`.
This is whole-body imitation of those retargeted locomotion references, not proof
of separate standing, upper-body or turning mastery. Labels do not prove all
motions are straight flat walking.

Completed656384 simulator transitions in1h03m50s training-loop time, final
logged WM loss.0872, logged8-step error.1859, bad gradients0. Logged scalar error
is not metres. Mean final-interval training episode27.91 controls=.5582s;
random-start training statistics are not fixed-test success. Final900→1000 audit
confirms finite changes in all network components and optimizer steps
WM9000→10000,actor900→1000,critic18000→20000. Projection checks over10256 batched
physical steps: max residual9.99983e-6, free momentum error1.0191e-7, energy
increase0. Audits are in `physics_v3_1000/{checkpoint_audit,training_summary}.json`.

## Fixed holdout and actual effects

Existing `independent_holdout_20261025/test.json`:7 accepted retargets from the
preselected12-source holdout;5 rejections remain, without replacement/balancing.
Existing near-limit advisory acceptance remains visible; not every clip passed
strict retarget-quality flags. Current train/test source-family and
mirror-normalized ID overlap are both zero. Three standing-upper and four
mixed walking-labelled clips include turning/bending/arm movement; not uniformly
flat walking. Evidence: `physics_v3_1000/evaluation_input_audit.json`.

One deterministic frame0 start per clip/agent, no DR/no restart after failure,
original termination gates, requested min(20s,clip length). Results:

| Clip | Requested s | Trained s | Zero-offset PD s | Trained stop |
|---|---:|---:|---:|---|
|003245|12.04|0.68|0.96|ee_body_pos|
|012584|12.12|1.04|0.90|ee_body_pos|
|003363|13.28|0.32|0.50|ee_body_pos|
|008312|20.00|0.46|0.68|ee_body_pos|
|005048|10.36|0.88|0.88|anchor_pos|
|004518|20.00|0.22|1.04|ee_body_pos|
|010195|20.00|0.42|0.64|ee_body_pos|

Both agents completed0/7. Trained mean/median.5743/.46s versus zero.80/.88s;
no improvement demonstrated. Several reviewed episodes visibly tilt/lose support;
others stop at tracking-error gates before impact. No20s drift comparison is
valid, and no direct comparison with the old PhysX PPO policy was performed.
No episode reached later full-motion turning/arm phases.

World-model diagnostics under recorded real actions reveal substantial free-base
prediction error. In the only three trained episodes long enough for H32(.64s),
root-position p95 errors are.170/.348/.250m and orientation p95
1.843/1.308/1.756rad; predicted joint-speed violations~18.9–19.2%. These are
learned predictions, not violations of the bounded physical simulator. Only37
overlapping H32 windows from three short episodes exist; other episodes are
explicitly unavailable, not padded. This supports prioritizing model validation,
not a proven causal explanation or production acceptance.

## Videos and verified delivery

Canonical results,28 front/side videos(7 clips×2agents×2views),raw state/action
traces, WM diagnostics, per-frame errors and color legend are in
`physics_v3_1000/evaluation/`. All28 pass H264/960×640/50fps/exact-step-frame
checks and full ffmpeg decoding. Front/side cameras follow translation but retain
initial world heading through turns. No apparent detached parts or whole-body
cropping in reviewed terminal frames for all14 episodes; temporal samples for
012584/003363 also inspected. This is sampled visual review, not every-frame
manual inspection. Audits: `video_audit.json`, `visual_review.json`, `summary.json`.

`trained_failure_reel_025x.mp4` includes all7 trained failures in fixed order,
front/side together, clearly labelled0.25x playback. Every original frame repeats
exactly4 times:804frames,50fps,1920×688,16.08s playback for4.02s of summed real
failure episodes. No fabricated continuation or terminal freeze. Reel audit and
whole-file decoding pass; unmodified real-time originals remain available.
Generator: `scripts/practice9/make_supermimic_failure_reel.py --evaluation PATH`.

`kinematic_reference/` contains14 full reference front/side videos, all-frame FK
max1.005e-7m, exact-frame/codec/full-decode checks, representative visual review
across all7 motions. **REFERENCE / NOT POLICY** writes qpos directly, without
physics integration or learned control; it cannot count as skill. Generator:
`scripts/practice9/render_supermimic_reference.py` with manifest/MJCF/output.
Original HumanML3D videos are linked for every motion, original20fps versus
retarget50fps; time-stretched robot references are not synchronized to them.

Current local SSH forward18768 serves data `datasets/practice9` through target
127.0.0.1:18768, tmux `tw_supermimic_videos`:
http://127.0.0.1:18768/supermimic_tw_20261031/physics_v3_1000/evaluation/index.html
This depends on the SSH forward/server remaining available. Server artifacts
remain on the data disk; they are not stored in Git or downloaded locally.

## Remaining research, not started automatically

1. Validate and improve the world model on sustained physically stable collected
   episodes, measuring root position/orientation/velocity and contact behaviour
   at H1/H8/H32 before further actor optimization. Large free-base error and
   prediction-only speed violations are the current evidence; mechanism is not
   established. More iterations alone are not proven to fix it.
2. Audit nominal-pose/action ranges versus reference poses and obtain suitably
   varied stable standing/small-step data before claiming those abilities.
   Any changed training distribution requires new, documented experiments.
3. Calibrate small-robot contact/perturbation/noise settings; operator-split speed
   constraints and mjlab dynamics are not exact PhysX equivalents.
4. Use multiple seeds, sufficient training and a controlled baseline on the fixed
   source-isolated holdout. Keep quality flags/early failures and all clips;
   do not change gates, cherry-pick checkpoints or auto-promote.

The environment and method trial are verified. Reliable SuperMimic control,
outperforming the existing PPO policy, long-horizon robustness, efficiency gains
and real-robot transfer remain unproven. Broader Notion004 stays in progress.
