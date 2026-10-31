# Real-feedback PPO learnability diagnostic (2026-10-31)

Project TW-RL / Notion004; ongoing, no promotion. Original target remains full-duration success on the fixed seven independent motions, at least three independent training seeds, matched-condition baselines and front/side videos. A single static diagnostic does not satisfy it.

## Evidence and reason

Native GPU PD/control mapping, force saturation, whole-mass velocity constraints, and warmstart probes have not identified an error sufficient to explain learned failures. Nominal zero-action PD stands for 20 seconds. The best learned nominal static result is only 1.68 seconds. Multiple world-model calibrations improved some diagnostics but failed other predeclared gates; all failures remain on the data disk. This experiment asks whether the same SuperMimic actor/critic and actual physics/reward can learn through real feedback. It neither replaces the SuperMimic acceptance target nor isolates one causal algorithm difference: real-sample budget and optimization differ.

## Fixed setup

- Parent `standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt`, SHA256 `92a3be6e562dd5f717bca1fed664da1d41a621a6f8cc7542d4a5e2c5994f0b1f`, iteration1000.
- Static synthetic default pose manifest SHA256 `7d010885950b444306e951b1b36e8255530041498e017775d641404ed0d74088`; motion SHA256 `6a43bbd638759e15523518aa2373b39785f1019d8d11b49b930c0a291a25a7a7`. No HumanML3D source; no independent-motion claim.
- Formal: 256 environments, seed42, exactly1000 PPO updates, rollout32, 8,192,000 actual transitions. Fresh initialization from parent; smoke weights are not used.
- Same encoder, decoder, critic and ten-frame history; zero only the final action readout; all30 action dimensions trainable. Actor/critic adapters verified exactly against nonzero parent outputs before zero initialization.
- Original training reset ranges, randomized friction/encoder/COM events, actual reward, original termination conditions and velocity/physics contracts remain enabled. Fixed Gaussian std0.1 from source, original sensor noise; no action clipping or head mask.
- Freeze source actor/goal/critic normalizers for valid PPO likelihood ratios. No world-model calls, fits or imagination during training; its source tensors stay unchanged. This intentionally differs from original online normalization/model training.
- PPO: Adam lr1e-4 fixed, four epochs/four minibatches, clip0.2, gamma0.99, lambda0.95, value weight1, clipped value loss, gradient norm1, entropy coefficient0 (fixed std).
- Timeout-only endings bootstrap terminal observation value before reset; true terminations do not. Done worlds alone reset, live states/context/clocks asserted unchanged; reset history padded with current proprioception rather than crossing episodes.

## Smoke and resume evidence

Data root `datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/` on the remote data volume.

`real_ppo_smoke_v1`: completed four updates but checkpoint audit initially failed because an Adam step scalar moved from CPU to CUDA on load. No numeric mismatch was established; failure log and checkpoint preserved. Device-independent value comparison fixes the check. Checkpoint verification now restores a separate optimizer, avoiding mutation of the training optimizer during saves.

`real_ppo_smoke_v2`: eight environments, original four-update total, stopped at update2, preserved `checkpoint_0002.pt` and `audit_interrupted_0002.json`, resumed actual PPO optimizer to update4. Final checkpoint SHA256 `2bcc4c7d513c5337a151940b337bd958a7aa77fe1dfdf8554ec34788051e5e13`. Final audit confirms exact parent adapter outputs, unchanged world model/normalizers, zero world-model forward calls, finite weights/losses, fixed std and optimizer roundtrip. Resume segment includes19 endings and3 timeout-only endings; live reset max state error0. These are interface checks, not standing success or bitwise simulator resume.

Checkpoint kind `real_physics_ppo_diagnostic_policy_v1` is rejected by normal SuperMimic continuation before loading any model weights. Evaluation must explicitly request `evaluation_only=True`. Only the diagnostic driver resumes its PPO optimizer; original SuperMimic optimizers are never serialized as if they were current. Simulator state and return sampler are rebuilt on resume, consistent with existing trainer limitations; this is not bitwise trajectory continuation.

## Run and gates

Run `bash scripts/practice9/run_supermimic_real_ppo_diagnostic.sh` inside tmux. The script takes a lock, recovers only explicitly completed diagnostic checkpoints, preserves incomplete evaluation for inspection, evaluates only final update1000, and records status without promotion.

At final update1000, evaluate original20-second window for learned and nominal-PD agents, stop on first original termination, save front/side H26450fps videos, actual traces and native contacts. Verify exact frame counts and full decoding, then inspect frames and compare actual common durations, root tilt/drift and actions honestly. No intermediate checkpoint selection. A completed nominal static episode would justify further perturbation/multiple-seed and small-step diagnostics, not full-motion acceptance. Failure requires examining actual rollout/control evidence rather than changing thresholds or replacing tests.

## Independent final artifact audit

`audit_supermimic_real_ppo_artifact.py` reads CPU tensors only and independently checks the supplied parent, manifest and each motion hash; exact fixed plan and real-transition budget; unique contiguous per-update metrics; finite serialized tensors/losses; absence of stale SuperMimic optimizers; all26 trainable parameter tensors' Adam moments and expected16 optimizer steps per PPO update; unchanged source world model and12 normalizer tensors. It refuses to overwrite prior evidence and rejects ambiguous duplicate/reordered logs instead of silently choosing an attempt. This verifies training artifacts only, not physical stability.

The preserved interrupted/resumed four-update smoke passes:1024 actual transitions,64 Adam steps for each of26 trainable tensors,12 unchanged normalizer tensors, unchanged world model, all source/manifest/motion hashes checked. All encoder/decoder/critic tensors differ from parent, though differences from parent include the declared readout initialization and do not measure skill.

Formal invocation after the final checkpoint exists:

```bash
python scripts/practice9/audit_supermimic_real_ppo_artifact.py \
  --run /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/real_ppo_standing_v1 \
  --parent /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt \
  --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json \
  --expected-updates 1000 --expected-envs 256 \
  --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/real_ppo_standing_v1/artifact_integrity.json
```

This requires8192000 new real transitions and16000 Adam steps per trainable tensor; the currently running training source snapshot stays unchanged. A separate CPU-only waiter can run this after the confirmed training PID exits, in parallel with the already arranged final evaluation.

## Final1000 result: nominal duration passes, tilted support rejected

Final checkpoint SHA256 `182fe0bdc89fbcdbfb38bde63b0c088903f9b71a4123d7813b176dd574202207`. Independent artifact audit passes8192000 new real transitions,16000 Adam steps on each of26 trainable parameter tensors,12 unchanged normalizer tensors and unchanged source world model. No world-model call was used for training. All1000 metric updates are unique and contiguous.

Both learned and nominal-PD agents complete1000/1000 steps (20s), without original terminations. All four front/side videos are H264/50fps/960x640,1000 frames, fully decoded. Twelve fixed sampled frames (start/mid/end per view/agent) were inspected; this is not an assertion of human inspection of every frame.

| Full actual20s | Learned | Nominal PD |
|---|---:|---:|
| Root horizontal displacement max |38.732mm|2.880mm|
| Root displacement at end |29.334mm|2.064mm|
| Root tilt max |8.780deg|0.669deg|
| Mean body position error |41.965mm|2.921mm|
| Mean joint angle error |6.074deg|0.054deg|
| Last10s left/right sole tilt median |13.105/24.314deg|0.421/0.426deg|
| Last10s loaded contact points each foot |2 in every frame|4 in every frame|

The policy widens its stance and persists on tilted foot edges. The duration improvement from the prior1.68s learned policy is real, but this is not accepted as reliable flat-foot standing and was not promoted. The nominal PD evidence contradicts a claim that the geometry can only support point contacts. Loaded point counts alone do not measure a continuous pressure area or hardware stability. Different real-sample budgets and optimizers prevent a matched algorithm superiority claim.

Artifacts are under the remote `longterm_diagnostics/real_ppo_standing_v1/evaluation`: `standing_metrics.json`, `standing_contact_summary.json`, `visual_review.json`, raw traces/contacts, montage images and `reviewed_index.html`. The reviewed page includes all30 joint-color legend entries, camera conventions, an explicit synthetic-source label and the PD comparison. Colors are baked offline from the previously saved precolored USD palette into a physics-identical MJCF. No runtime material binding or HumanML3D-source claim is made.

## Frozen-policy factorial follow-up

`collect_supermimic_learned_factorial.py` separates original reset perturbations, source-amplitude uniform sensor noise, and Gaussian action noise in a2x2x2 factorial, for learned and PD agents,32 worlds each. Every case starts at reference frame0 and records only its first original20s episode, stopping on original terminations. Randomized events are disabled to isolate these factors, so this does not reproduce the full training distribution. The same policy seed42 is used; this is not three independent training seeds or independent-motion validation.

Fixed action draws use NumPy seed10042, Gaussian std0.1. Sensor draws use seed20042, independent uniform component amplitudes0.01/0.5/0.2/0.05, with zero noise on previous-action features. The source proprioception layout is preserved. Fixed arrays do not consume simulator/reset random numbers; all cases of the same reset factor assert matching initial state/context. Ended worlds are parked without resetting or modifying active worlds' state/context/clocks, and only first-episode prefixes are saved. Actor/normalizer tensors remain exactly frozen, the world model is never called, speed caps and finite/contiguous traces are checked.

The initial smoke caught command clock1 after reset. Its logs/configs remain in `smoke_v1_failed_clock`; the revised collector first verifies that the reference is constant with zero velocities, then standardizes only the clock to0 without changing perturbed physical state. The second smoke (4 worlds,40 steps,two extreme profiles,two agents) passes16 finite contiguous first-episode traces and active-world parking assertions. Nominal initial state matches the previous single-world replay exactly; first action differs by6.6012e-5 under batched floating-point inference, which is recorded rather than claimed bit-identical. Smoke duration is0.8s and is not counted as20s acceptance.

Run the smoke/formal modes through `run_supermimic_learned_factorial.sh`. New caches and artifacts stay on the data disk. Do not restart completed output directories or choose a checkpoint based on intermediate results.

## Factorial512 results and numerical pairing limitation

Each entry is completed20s first episodes out of32; policy seed42 only, no random events. R=initial reset perturbations, S=uniform sensor noise, A=Gaussian action noise.

|R/S/A|Learned|PD|
|---|---:|---:|
|000|32|32|
|001|22|0|
|010|32|32|
|011|26|0|
|100|18|0|
|101|4|0|
|110|17|0|
|111|6|0|

The archive's512 finite contiguous traces, initial state/context matching, exact PD action-noise application, speed limits and completion accounting pass independent checks. The strict PD sensor-toggle full-state atol1e-5 check fails and remains explicitly failed: `independent_integrity.json` has `passed=false`, `archive_invariants_passed=true`, `numeric_pairing_passed=false`. Maximum root position difference is9.298e-6m and root angular-velocity difference1.6725e-4rad/s, with identical completion/termination steps. The original failed verifier source and report are preserved.

A separate128-episode same-condition PD repeat has identical initial states/actions and ending steps; maximum root position difference4.813e-6m and angular-velocity difference1.2112e-4rad/s are of similar order. One repeat does not establish a statistical error bound or bitwise reproducibility. Sensor-noise benefit is not inferred from22 versus26 completions. The first repeat attempt had incorrect declared agent metadata and was interrupted/preserved as `repeat_pd_v1_failed_metadata`; the corrected repeat is separate. Original512 episodes were not rerun or changed.

The learned policy has some recovery capacity but remains unacceptable for tilted edge support and incomplete robustness. The next single training change is a reference-sole-normal penalty, with all other settings fixed; the physical termination gates are unchanged. None of these developer diagnostics fulfills the final fixed7-motion/three-training-seed target.

## Predeclared reference-sole-normal penalty experiment

The next training variable is `--sole-normal-penalty 2`. All other original real-feedback PPO settings are fixed: same original SM parent92a3be6e..., fresh final-readout zeroing, source normalization tensors frozen,256 worlds,seed42,1000 updates,8192000 actual transitions, fixed std0.1, original sensor noise/reset/events/physics/terminations. The smoke checkpoint is not a formal training parent. There is no joint masking, new degree of freedom, geometry or mass edit.

For the two feet, the new reward is

`base_reward - 2 * dt * mean_feet(||normal_actual - normal_reference||^2)`.

Both normals are signed world-space local-Z axes of the sole boxes. The constant synthetic reference is explicitly checked; the feature is not silently applied to general motion training. Native compiled model checks establish that each box is owned by its expected foot body with identity local rotation. Every sampled step compares the quaternion-derived normal with actual GPU geom_xmat (atol1e-6). Squared signed-normal distance distinguishes an upside-down sole from a correctly oriented one, remains yaw-invariant for a vertical reference and avoids acos gradients. Reward shaping also affects the existing return sampler through actual episode returns; the sampler algorithm itself is unchanged.

At the old13/24deg stance the penalty is approximately0.0046 per step, about7.5% of the observed mean base reward. This is a fixed initial coefficient, not selected using the new final result. It does not alter physical termination or evaluation gates. The default coefficient0 keeps the original reward tensor itself, without a sole computation.

`supermimic_sole_normal.py` verifies identity/yaw cost0,90deg cost2,180deg cost4 and finite gradients. The eight-world/four-update smoke was interrupted at update2 and resumed with the actual PPO optimizer to update4; both checkpoints and segment audits are preserved. Native normal error is at most2.9802322e-7, reward composition and penalty/cost scale pass independent checks, all26 Adam states have64 steps,12 normalization tensors and world-model tensors remain unchanged. Its final checkpoint SHA256 is `6824fc04d2611fdb48d535271468f718ccc87fa6ca963093dd0fd7ca9a080921`. The first pre-update rollout's base reward differs from the old coefficient0 smoke by only3.49246e-10, with8 endings in each; this supports unchanged initial sampling without claiming full GPU bitwise determinism.

Use `run_supermimic_sole_normal_diagnostic.sh smoke`, `resume_smoke` and `formal`. Formal output is `longterm_diagnostics/sole_normal_ppo_v1/trained`. The runner locks the run, preserves completed checkpoints for recovery, audits the final training artifact, evaluates only final update1000 for learned/PD20s with front/side colors and native contacts, and verifies exact video frames/full decoding. Physical contact, drift, pose and recovery still need review; no promotion follows artifact verification automatically.

Compare the final candidate with the preserved coefficient0 baseline on original20s gates and on the same R/S/A factorial, retaining all groups and the strict numerical-repeatability caveat. Static foot orientation improvements alone do not prove full-motion tracking, world-model consistency or three-training-seed acceptance.

## Coefficient2 final result and matched recovery diagnosis

Final checkpoint SHA256 `9c8569f1adcaf31982235c26c9be0864c2c1e9dd04193e67ea3821cd7e95957b`. All1000 updates/8192000 transitions and optimizer/frozen-normalizer/world-model checks pass. Quaternion normals agree with native GPU sole-box rotations to4.1723251e-7. Learned and PD each complete20s without original termination; all four videos are H264/50fps/960x640/1000 frames and fully decoded. Twelve fixed sampled frames were inspected, not every frame.

|Full actual20s|Coefficient0|Coefficient2 squared|
|---|---:|---:|
|RootXY max|38.732mm|18.822mm|
|Root tilt max|8.780deg|5.195deg|
|Mean body error|41.965mm|24.063mm|
|Mean joint error|6.074deg|4.396deg|
|Last10s sole tilt median L/R|13.105/24.314deg|7.555/4.912deg|
|Loaded points each foot, every last10s frame|2|2|
|Per-foot contact hull area ratio median|0|0|

Tilts and tracking improve, but both feet persist on an edge. The current PD has3 non-collinear loaded corners per foot with hull area ratio~0.5 and~0.42deg tilt; the earlier PD has4/~1 with nearly identical pose/drift. Preserve this contact allocation variation; positive-point hull area does not measure continuous pressure. No promotion or flat-foot acceptance follows.

The matched512-episode factorial uses bit-identical noise arrays and exactly matching initial states, with no PD ending differences across runs:

|R/S/A|Coefficient0 learned|Coefficient2 learned|PD|
|---|---:|---:|---:|
|000|32|32|32|
|001|22|29|0|
|010|32|32|32|
|011|26|26|0|
|100|18|22|0|
|101|4|13|0|
|110|17|23|0|
|111|6|12|0|

All counts are out of32 in one fixed developer protocol, policy training seed42 only. Archive invariants pass; strict atol1e-5 numeric sensor-toggle pairing remains failed (candidate maximum1.14083e-4), as explicitly recorded. This is not population significance, three-training-seed or independent-motion acceptance. `factorial_comparison.json` retains every group and the caveat.

`review_supermimic_standing_evaluation.py` generates metrics, native contact convex-hull areas, fixed-frame montages and the colored front/side review page with30-axis legend and camera/source conventions. Known square/triangle/line areas are checked. `compare_supermimic_factorials.py` verifies fixed inputs/noise/initials before reporting all groups. Neither tool promotes a policy.

### 2026-10-31 22:33 CST: smooth-distance single-variable probe

The coefficient-2 squared-normal run remains an unaccepted standing diagnostic:
20 s nominal survival, but both feet retain two-point edge support. Walking and
full-body imitation are not established. A new pre-registered probe changes only
the normal cost to d2 / (sqrt(d2 + 0.01**2) + 0.01), keeping coefficient 2,
source parent, seed, real-physics PPO budget and all existing termination gates.
The smoothing constant is not an acceptance tolerance. Four-update smoke and
optimizer resume passed independent artifact checks (1024 real transitions,
26 Adam states, frozen world model and normalizers); native-normal agreement
was 3.5762786865234375e-7. Formal results are pending. This is not a SuperMimic
world-model success or a replacement for fixed-seven, three-seed validation.

### 2026-10-31 22:35 CST: dynamic reference audit

All 332 training references were audited using the current MJCF sole box and
stored body transforms. Sampled native FK foot positions agree within
5.945e-8 m; fixed-seven IDs are excluded. 331 clips have both soles exceed
10 mm clearance at some frame. Across 663 foot records with corners within
2 mm of ground, the median per-record 95th-percentile tilt is 18.590 degrees
(maximum 47.163); 457 exceed 15 degrees. These descriptive bins are not gates.
Near-ground samples include toe-off and landing: this does not establish a
retargeting defect or dynamically feasible walking. Next inspect sustained
support and original source semantics before preparing a walking curriculum.
Script: scripts/practice9/audit_supermimic_training_soles.py; evidence remains
on the data disk in longterm_diagnostics/training_sole_clearance_audit.json.

### 2026-10-31 22:38 CST: sustained near-ground reference screen

The v2 audit retains the original clearance report and writes a separate
training_sole_support_audit_v2.json. A descriptive quiet-foot screen requires
corner height <=2 mm, finite-difference foot-origin speed <=0.03 m/s and
angular speed <=0.5 rad/s. Six single-foot records have >15-degree tilt for
at least 0.2 s: 004487 L .32 s, 006328 L .38 s, 006603 L .36 s,
011056 L .30 s, 012325 L 1.98 s, 013033 L 1.30 s. All original clips remain.
These are kinematic indicators, not force-based contact labels or new gates.
Prior near-ground tilt counts must not be equated to sustained bad support.
All 332 source-caption files were located and their hashes saved in the remote
training_motion_source_inventory.json; inherited walking labels alone do not
prove pure walking. Inspect the six segments and source videos before deciding
whether reference correction is justified. No walking-policy claim is made.

### 2026-10-31 22:47 CST: antithetic response fit prerequisites

The remote antithetic_response_fit_v1/plan.json pre-registers matched control
and candidate fits: same H8/H1 base losses, parent, data, 1000 updates and
forward budget; response-MSE coefficient 0 versus 1 is the sole variable.
Root angular and joint velocity response groups use immutable parent state
normalization scales. No optimization has started. Existing trajectory H1/H8/H32
and held temporal branch gates remain unchanged; static calibration is not proof
of walking dynamics. Normalizers, termination and policy remain frozen.
The new fit-pair auditor passed all 240 axis conditions (960 branch endpoints),
including fit-file hashes, sign swaps, action-axis order, finite targets and
no branch terminations. Maximum reset replay error was 4.47343e-6, below the
original collection tolerance 1e-5. No validation data were used in this audit.

### 2026-10-31 22:50 CST: response loss smoke completed

Implemented optional response loss while preserving the original branch fitter
when no response flag is provided. Both coefficient-0/1 four-update smoke arms
completed. Independent artifact inspection confirms 22 non-network tensors
unchanged, eight predictor tensors changed, finite weights and Adam step 4.
Smoke artifacts have a distinct non-policy kind and cannot imply full calibration.
The final-only 1000-update matched runner includes the existing trajectory and
branch auditors and frozen gate comparisons. Neither model is promoted.

### 2026-10-31 22:55 CST: response fit and frozen integration results

Final response fitting improved held root-angular action-response cosine from
0.73904 to 0.99760 and RMSE 1.14447 to 0.11928; joint velocity response RMSE
0.07621 to 0.03865. It nevertheless failed 11/18 trajectory checks vs parent,
13/18 vs matched control, and 6/9 branch checks. H32 root position p95 median
rose to 79.28 mm (parent70.26, control66.24). No promotion.
Frozen velocity-derived pose integration then reduced sole-normal response
RMSE 0.03561 to 0.007106 (cosine0.89594) and sole-cost RMSE0.0004184 to
0.0001326. Still failed 8 trajectory checks vs initial/control, 4 vs unwrapped
candidate, and branch absolute root-linear/joint velocity checks. This is an
explicit diagnostic prediction contract, not deployable raw-weight success.
Evidence remains in antithetic_response_fit_v1 and antithetic_pose_integration_v1.
The next hypothesis is to fit the pose-velocity structure during optimization
alongside the same response loss, changing only prediction structure.

### 2026-10-31 22:58 CST: fit explicit pose structure with response loss

The response_pose_fit_v1 preregistration changes only whether pose derives
from velocities during fitting. Response coefficient1, source parent, fit data,
seed42, 1000 updates and samplers stay fixed. The previous response candidate
(be0b85e7...) is reused as the exact declared control, never checkpoint-selected.
Four-update structured smoke passed: 22 frozen non-network tensors identical,
eight predictor tensors updated, finite weights, four Adam steps, explicit
fit-only prediction contract with 4505 fit frames. Smoke retains a separate
non-policy artifact kind even when prediction metadata are present. Formal
structured artifacts require their explicit prediction wrapper; raw weights
alone are not equivalent. Original all-horizon and branch gates remain fixed.

### 2026-10-31 23:04 CST: structured-response fit rejected; loss audit

Structured candidate29a82c79... failed5/18 trajectory checks vs initial H8,
4/18 vs the reused response-only candidate, and3/9 branch checks. The latter
include root-linear absolute error plus insufficient additional20% velocity
response improvement; velocity response itself did not regress. Counts against
different controls must not be compared as the same improvement measure.
The same initial-H8 comparison changed from11 failures (response-only) to5.
No promotion. Existing response_pose_fit_v1/review.json retains all failures.

Extended fit-only loss decomposition to accept explicit compared checkpoints
and include all240 antithetic conditions. Structured model objective consists
of baseH8/H1 loss3.683%, root response63.390%, joint response32.927%.
This identifies objective-scale imbalance as a hypothesis, not proof that
changing weights will pass. Next measure parent fit-only scales before fixing
any coefficient; do not tune against validation. The first audit wrote its
complete report then failed in console formatting of the new row schema;
that log/report are retained. The corrected v2 audit completed independently.

### 2026-10-31 23:09 CST: fit-only coefficient measurement

A frozen initial-H8 parent with the same explicit pose structure was measured
on all3249 fit H8 windows,1200 H1 transitions and240 paired response conditions.
No optimizer or validation selection ran; all weights and normalizers match
source. Base loss4.7231721401e-5 / response loss0.154362171888 yields the single
predeclared coefficient0.0003059798966495334. The rule equalizes initial parent
loss magnitudes, not validation thresholds or gradients. No coefficient sweep.
The coefficient and fit sources are cryptographically bound to the source
measurement, parent, integration audit and pair audit in the calibration plan.
A deliberate doubled coefficient was rejected before optimization; that expected
failure is retained. Four-update smoke passed frozen tensors, finite weights,
optimizer steps and immutable control hash checks. Its loss is not acceptance.
Formal response_balance_fit_v1 changes only coefficient vs the structured
coefficient1 control29a82c79..., with original data/samplers/1000 budget/gates.

### 2026-10-31T15:15:47+00:00 : rejected balanced fit and coverage diagnosis

The measured-weight candidate de9b9d4b... failed6/18 trajectory gates against
both initial parent and the coefficient1 structured control, plus5/9 branch
gates. H32 root-position p95 median83.08mm vs control64.12mm. No promotion.
Held-branch root-linear RMSE improved3.301 to1.375mm/s, but root-angular
response RMSE worsened0.1183 to0.3596. Weight balancing did not solve rollout.

Read-only state-coverage audit evaluated both models on all existing static fit
and temporal-validation trajectories/branches, preserving weights and norms.
In held branches, common root-linear bias share dropped76.3% to19.0%; trajectory
validation share39.1% to1.4%. Normalized nearest-fit input RMS p95 was0.002057
for branches and0.08502 for trajectories. These descriptive distances do not
prove adequate coverage or independent motion generalization. Stop coefficient
trials; next inspect state-response sensitivity and autoregressive error growth.
Evidence: response_balance_fit_v1/review.json and response_state_coverage_v1.

### 2026-10-31T15:20:33+00:00 : root feedback oracle diagnosis

Read-only oracle feedback ablations cover all4222 H32 windows across eight
existing temporal-validation traces for both structured and measured-weight
models. First-step predictions are identical across all modes; weights and
normalizers unchanged. Replacing previous-frame root linear velocity with
truth reduces median trace H32 position p95 from64.12/83.08mm to2.919/4.379mm;
root angular velocity replacement gives18.789/19.253mm. Joint velocity gives
60.775/78.628mm. These are oracle hybrid states, possibly physically
inconsistent, not deployable models or evidence of real control success.
Results identify root velocity feedback as a priority for native state-response
measurement, not a proven physical cause or permission to replace predictions
with truth. Existing mjlab root-link velocity writer was inspected: it accepts
world-frame linear/angular velocity, matching the73-state extractor, and
converts angular velocity into native free-joint body coordinates internally.
Next use paired native root-velocity perturbations under the original physics
contract and terminations, with declared temporal fit/validation partitions.
No such native collection has started yet. Evidence: response_feedback_error_v1.

### 2026-10-31T15:23:56+00:00 : native root velocity branch smoke

Implemented native root-link velocity branches under the original v3 physics,
zero-action PD and termination rules. Twelve worlds cover six root velocity
axes with antithetic signs and swapped assignments. Linear deltas0.005/0.010
m/s and angular deltas0.020/0.040rad/s are written in world coordinates;
all other73-state components and context are independently checked unchanged.
Smoke warmup5 collected60 one-step transitions, no terminations. Independent
archive audit passed: maximum requested-state error1.118e-8, reset replay
error9.909e-7 below original1e-5 tolerance. These are state perturbations,
not action perturbations or episode windows. Full collection uses fit warmups
5/25/75/150 and temporal-validation15/35/125/175, declared before simulation.
The diagnostic does not establish motion-source generalization or train policy.

### 2026-10-31T15:29:29+00:00 : native state sensitivity evidence

Full native collection completed40 files/480 transitions, no terminations;
requested-state error1.49e-8 and replay error3.62e-6 passed independent audit.
Frozen root-state response comparison includes all declared conditions and
three fixed models, without fitting. Held root6x6 velocity-subblock spectral
radius median is0.91723 native,0.98972 initial H8,0.98722 structured response,
and0.97827 measured-weight. These are partial local derivatives, not whole
system or closed-loop stability proofs. They support investigating missing
state-response damping supervision, distinct from action-response training.
V2 reports response errors in declared normalized input coordinates; v1 raw
coordinate summaries are retained. Raw matrices and spectral results agree.
Preregistered root_state_response_fit_v1/plan.json compares coefficient0/1 for
added root-state Jacobian loss with identical previous terms and forward budget.
No implementation or fitting has started; smoke and source audit remain required.

### 2026-10-31T15:33:47+00:00 : root-state response smoke

Implemented root-state response supervision using fit48 axis conditions only.
Central differences match the independent native float64 calculation; input
and output Jacobian coordinates use immutable parent state standard deviations.
The gradient is finite/nonzero(norm0.140855). Both four-update smoke arms passed
independent frozen22-tensor, updated8-tensor, optimizer-step4 and artifact-kind
checks. No policy artifact or prediction acceptance is implied.
The formal matched pair keeps H8/H1/action response and explicit pose structure,
adds64 root-state endpoint predictions equally in both arms, and changes only
root-state coefficient0/1. Metadata now explicitly separates576 base full-state
targets from704 prediction endpoints per update; original target count omitted
paired endpoints as full-state targets because their loss is a Jacobian loss.
The full runner retains all original audits and adds all native root-state
conditions without introducing a new relaxed gate or automatic promotion.

### 2026-10-31T15:43:51+00:00 : corrected damping but biased short-horizon prediction

Root-state supervision corrected held local velocity-subblock radius to0.91565
(native0.91723, matched control0.98722). All six H32 nonregression checks pass
vs both initial parent and matched control: root-position p95 median59.93mm
vs70.26/64.12mm, root-orientation0.28735rad vs0.29428/0.32042rad.
Overall still rejected:4 trajectory failures vs parent,5 vs control,9 branch
failures. Velocity action derivatives remain close to control but do not meet
another20% improvement requirement; short-horizon absolute errors also fail.

Fit-only root-linear/root-angular/joint velocity common-bias fractions are
83.9/89.3/76.1%; temporal-validation83.5/87.0/72.3%. A preregistered next
post-fit diagnostic will subtract the unique fitH1 mean velocity error over all
1200 original fit action-branch transitions, altering only36 output bias entries.
No optimizer, coefficient search or validation fit; all other weights remain
frozen. Verify Jacobian preservation and all unchanged gates before conclusions.
This operation is planned, not implemented, and is not another1000-update fit.
Evidence: root_state_fit_coverage_v1 and velocity_mean_correction_v1/plan.json.

### 2026-10-31T15:53:27+00:00 : analytic mean correction rejected overall

Analytical36-output velocity mean correction used all1200 original fit branch
transitions, no optimizer or validation fitting. Only final velocity biases
changed; other weights/norms and pose contract preserved. No stale optimizer.
The source1000-update counter is provenance, not new optimization. First strict
unwarmed test found quaternion repeat rounding8.94e-8 before modification;
warmup resolved it without relaxing equality checks. All failures retained.
Final metadata clarification has bitwise identical corrected weights.
All six held-branch absolute state checks now pass. Root-linear RMSE4.282 to
1.821mm/s, root-angular0.01711 to0.006301rad/s, joint0.008815 to0.004759rad/s.
Overall rejected:4 trajectory gates vs initial and source, plus3 unchanged
velocity/sole-response additional-improvement gates. H32 root-position p95
median59.23mm; fixed correction does not solve state-dependent error.
Evidence: velocity_mean_correction_v1. No policy promotion.

### 2026-10-31T15:58:35+00:00 : conditional velocity anchor smoke

Implemented preregistered conditional velocity-value anchor. It equally averages
parent-normalized root-linear, root-angular and joint velocity MSE, half from
existing H8 predictions/targets and half from H1. No additional forward passes
or true future states as model inputs. Existing H8/H1 and both derivative
losses, pose contract, data and samplers are unchanged; coefficient0/1 only.
Both four-update smoke arms passed frozen22 tensors, updated8 predictor tensors,
Adam step4, fit-only root-state sources and704 prediction endpoint metadata.
This is validity evidence only; formal1000 update results remain pending.
Standing real-physics PPO independently reached847/1000; no interim selection.

### 2026-10-31T16:09:10+00:00 : velocity anchor result and root-pose diagnosis

Conditional velocity-anchor final1000 fit remains rejected:3 trajectory gates
vs initial parent (H1 root angular velocity, H8 root orientation/angular velocity),
3 vs matched control, and3 additional-improvement branch gates. All six held
branch absolute-state checks pass. Candidate3c6c21b3955a72127e2e6a4af9c55e9917d28e5c59808ce4eb1c070a12d514df.
The source collection was re-read: events disabled; forcing is recorded30-axis
sinusoidal action only, no unrecorded external force injection was found there.
This does not prove every possible hidden simulator state is irrelevant.
Fit-only quaternion input scale over4505 states is std[w,x,y,z]=
[0.0060222,0.0023550,0.0337446,0.0207831]. Frozen quaternion features therefore
have quite different scales, a conditioning hypothesis to test, not a proven bug.
Preregistered native root-pose6-axis probe will preserve world root velocity
when changing orientation (rewrite velocity after pose), all joint state, and
original physics/terminations. Deltas0.5/1mm translation,1/2mrad rotation,
fixed existing temporal partitions and swapped signs. No collection started.
Evidence: velocity_anchor_fit_v1/review.json, quaternion_input_scale.json,
and root_pose_branches_v1_protocol/plan.json. No policy promotion.

### 2026-10-31T16:11:53+00:00 : native root-pose smoke verified

Implemented root-pose interventions with world rotation left-multiplication,
position displacement, and explicit rewrite of unchanged world root velocities
after pose update. Joint state/context remain fixed. Translation0.5/1mm and
rotation1/2mrad, all six axes and swapped signs, original gates retained.
Smoke60 native transitions passed independent NumPy quaternion composition,
state/context/archive checks, no terminations: requested-state error1.016e-7,
replay error1.621e-7. This is native response data, not policy success.
The formal fixed temporal fit/validation collection is ready; all contact-mode
changes will be retained. Standing real-physics PPO reached977/1000.

### 2026-10-31T16:29:25+00:00 : final standing video review and native position dependence

Smooth-distance real-PPO final1000 completed and independently audited:
checkpoint505ea53d443f394e5c8182c993b61d87170473f4c49efc53d7a3f8374345362c,
8192000 real transitions, frozen world model/norms. Learned and zero-action PD
both complete20s. Four colored front/side H264 videos each1000 frames pass full
FFmpeg decode. Actual six-frame montage review per agent: upright, visible parts
attached, learned torso inclines relative to PD, extended-arm reference remains.
Learned rootXY max8.623mm, tilt max6.985deg, body mean16.362mm. Last10s left
8.514deg/two loaded points500frames/area0; right0.621deg/four points500frames/
area ratio0.99994. Persistent left edge support is not accepted. Not walking,
not SuperMimic success; synthetic reference has no HumanML3D source.
Review page on forwarded18768 verified HTTP200. Fixed R/S/A factorial next,
same32 worlds, original noise arrays and all eight conditions, final model only.

Native root-pose collection480 transitions passed independent audit, no terms;
max state-write1.69e-7/replay4.82e-6. Frozen models show spurious XY-dependent
angular response: latest0.000498/0.000811rad/s per declared0.5mm input scale,
native roughly1.63e-7/4.52e-8. Do not headline relative ratios near zero.
Preregistered translation_invariant_fit_v1 will enforce dynamics-only horizontal
translation equivariance by zero first-layerXY columns with reference-origin
bias compensation and gradient masks, no state-format or inference-hook change.
All normalization/termination parameters remain frozen and original gates fixed.
No fitting started; initial-function, smoke and tensor checks still required.

### 2026-10-31T16:35:10+00:00 : horizontal symmetry smoke

Dynamics-only XY equivariance implemented as zero first-linear input columns0/1,
reference-origin bias compensation and matching gradient mask; ordinary exported
weights enforce the property without inference hooks. Positional outputs retain
actual rootXY and term-net positional-goal dependence is untouched.
Both four-update smoke arms pass frozen norms/termination, Adam-step4,704
endpoint metadata and candidate zero-column checks. Initial reference function
max difference2.384e-7; constructed horizontal-translation property error1.863e-9.
This is not prediction or control acceptance. Same H8/H1, action/state Jacobian
and velocity-value losses; formal1000 matched pair keeps all original gates and
reports all native pose/velocity responses. No policy promotion.

### 2026-10-31T16:43:37+00:00 : horizontal symmetry result and conditioning plan

Horizontal-equivariant final1000 candidate passes all18 trajectory nonregression
checks against its matched control, but still fails3 vs initial H8 (H1 angular
velocity, H8 root orientation/angular velocity) and4 branch checks. Rejected.
Native-pose validation confirms XY angular responses reduced from control
0.000498/0.000811rad/s to1.763e-7/4.526e-8, matching native noise-scale response.
This fixes a spurious position dependency, not complete model/control validation.

Preregistered next quaternion feature-conditioning experiment keeps physical
state/outputs, non-network normalization, existing losses and XY symmetry fixed.
Fixed mean/std use4505 original fit states only. A training-only affine feature
pre-hook plus first-layer compensation must preserve the initial function;
remove hook and algebraically export ordinary raw-coordinate weights. Verify
round-trip inference. Conditioned-coordinate optimizer must be labelled or
removed rather than represented as ordinary raw-coordinate resume state.
No implementation started. Evidence: quaternion_conditioning_fit_v1/plan.json.

---
### 2026-10-31T16:49:28.156827+00:00 | Smooth-normal factorial review

All 512 first episodes / 16 profile-agent groups archived; archive invariants pass, strict PD sensor-toggle pairing fails at original 1e-5 tolerance (max state difference 0.0006588585674762726). No tolerance change. Fixed input/noise arrays match the squared-normal baseline; PD endings match.

Trained 20s completions, baseline -> candidate, each /32: R0S0A0 32->32; R0S0A1 29->26; R0S1A0 32->32; R0S1A1 26->28; R1S0A0 22->20; R1S0A1 13->13; R1S1A0 23->22; R1S1A1 12->9. Do not infer sensor-noise benefit from these pairs. Candidate rejected as a general improvement; not promoted. Raw traces/checkpoint retained. Final-motion and multi-seed acceptance remain incomplete.

Evidence: `/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/sole_distance_ppo_v1/factorial_comparison.json`, `/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/sole_distance_ppo_v1/factorial_review.json`. Next: test predeclared quaternion feature-conditioning with raw-coordinate export and unchanged physical/model gates.

---
### 2026-10-31T16:52:33.343638+00:00 | Quaternion export audit

Added read-only raw/conditioned first-layer export reconstruction audit. Smoke artifact checked against all 4505 original fit states with source hashes verified: max forward difference 7.152557373046875e-7, below unchanged 1e-5 export tolerance. This is serialization equivalence only, not held-out accuracy/control acceptance. Evidence: longterm_diagnostics/quaternion_conditioning_fit_v1/smoke_export_all_fit.json. Formal control completed 1000 updates and entered independent integrity/trajectory audits; candidate follows in the existing tmux runner.

---
### 2026-10-31T16:55:13.139278+00:00 | Quaternion conditioning rejected

Both arms completed 1000 updates and all prescribed audits. Candidate SHA256 64442c67001c5170737c40255c25e5b9df63ea67d2545ac9113404bb1466bcaa. Export equivalence on all 4505 fit states passes (max 9.8347664e-7), but this does not establish accuracy. Matched trajectory gate: 5/18 failures; initial-parent gate: 4/18 failures; branch gate: 7/9 failures (four absolute state errors and three derivative improvement requirements). H32 orientation p95 median 0.27875248 -> 0.29743601 rad (+6.70%). H8 root linear RMSE 0.01470049 -> 0.01613753 m/s (+9.78%). No promotion or coefficient search. Evidence: quaternion_conditioning_fit_v1/review.json, original gate JSON and root-pose/state response audits retained. Next investigation: direct native root-pose-to-velocity response mismatch; feature scaling alone is not supported as a remedy. Full-motion/multi-seed gates remain incomplete.

---
### 2026-10-31T16:57:22.075568+00:00 | Native pose-response supervision

After rejection of quaternion conditioning, add fit-only root-pose response loss coefficient0/1 with same 64 extra endpoints in both arms (768 total per update). Normalize physical central response by declared base_delta and frozen parent velocity std, not near-zero native relative error. No quaternion conditioning; preserve XY invariance and prior loss/data/parent. Independent float64 target check max error1.19648308e-6 <=2e-5; gradient finite nonzero0.0004389913. Two 4-update smoke arms pass frozen non-net parameters, zero XY columns, optimizer steps and source contracts. Smoke ran before a metadata-only differing_variable description correction; arithmetic unchanged. Fixed1000 final-only fits and unchanged original audits follow. No policy promotion. Evidence: root_pose_response_fit_v1/plan.json, response_loss_probe.json, smoke_integrity.json.

---
### 2026-10-31T17:01:34.863463+00:00 | Pose-response trial rejected; variation audit

Root-pose supervision candidate completed1000 with original gates: paired trajectory3/18 fail, original-parent3/18 fail, branch5/9 fail. H32 root position p95 median58.432->63.471mm. Candidate SHA2562dd8556a7b3b1945066afa07963e0e2644e9becfa729504a3a6dd6387a994774; rejected, no promotion.

Read-only response-variation audit across all fixed conditions: validation vertical-input angular response native condition std0.00110959rad/s, candidate0.00004324rad/s. Candidate full response RMSE0.00328670rad/s vs control0.00514270rad/s; a fit-only constant-response diagnostic gives0.00134944rad/s. This is evidence of insufficient predicted response variation, not proof of the cause or deployable correction. Preserve fit-only mean, never fit validation. Next inspect state-dependent/contact-dependent representation and fit coverage before another gradient experiment. Evidence: root_pose_response_fit_v1/review.json and response_variation.json. Final independent motion/multi-seed requirements remain incomplete.

---
### 2026-10-31T17:05:13.270977+00:00 | Native pose geometry boundary audit

Read-only CPU native kinematics examined all480 root-pose branch input poses. Initial robot-asset plane assertion failed because terrain is external; no report was produced from that failed attempt. Verified TW inherits plane terrain and installed importer creates default world plane at z0; explicit required ground-height argument now records this and verifies any embedded plane. Source hashes saved in ground_configuration_provenance.json.

Baseline minimum foot corners are around -0.49 to -0.56mm. At level2 (+1mm root height), the eight fixed warmups have lifted-side minima +0.444 to +0.508mm with zero corners below plane. At level1 (+0.5mm), seven of eight warmups retain 2-4 corners below plane; warmup5 has zero. Therefore these finite differences can cross geometry regimes and must not be described as universally infinitesimal Jacobians. Geometry does not establish force/contact loading or causality. Evidence: contact_geometry.json and plane_crossing_review.json. Next compare exact differentiable foot-clearance features against native geometry and inspect coverage before model changes; do not alter physics, samples or gates.

---
### 2026-10-31T17:07:02.136869+00:00 | Differentiable sole clearance verified

Implemented reusable native-asset-derived box corner height feature, keeping robot state73 and physical geometry unchanged. CPU native versus differentiable GPU FK verified for all480 archived pose branch input states: maximum height error3.9144652e-8m below fixed1e-6m. Every state passed horizontal translation invariance, quaternion-sign invariance, finite gradients and summed root-height gradient16 (16 corners). No force inference, contact labels, runtime policy modification or acceptance claim. Evidence: root_pose_response_fit_v1/differentiable_clearance_audit.json. Next audit fit/validation clearance coverage before defining a model feature experiment; no default policy promotion.
