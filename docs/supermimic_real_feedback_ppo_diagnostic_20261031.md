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

---
### 2026-10-31T17:09:15.755425+00:00 | Clearance coverage audit

Source-hash-verified fit groups:3305 trajectory states,1200 action branch states,240 root-velocity-response states,240 root-pose-response states. All240 held root-pose input states queried without parameter fitting. Nearest16-corner RMS height distance p95: trajectory0.55334mm; action branches0.97091mm; root-velocity branches0.97091mm; root-pose branches0.035640mm (max0.036052mm). The pose-response fit source already includes nearby geometry, weakening a wholly-unseen-clearance explanation; does not establish full state/action/velocity/contact-force coverage.

Evidence: root_pose_response_fit_v1/clearance_coverage.json. Next predeclare a geometry-feature model comparison preserving original state73/physical system and initial predictions, with explicit artifact/loader contract, matched parameters/budget, fit-only statistics and unchanged H1/H8/H32/branch gates. Do not use held validation for normalizer fitting. No new training or promotion from this audit.

---
### 2026-10-31T17:13:13.694665+00:00 | Explicit offline geometry model contract

Added16 features at the dynamics first layer: eight local-bottom box corner signed heights and negative-height ReLU, scaled by fixed1mm. Both arms have identical added parameters and FK computation; control multiplies features by0, candidate by1. Added first-layer columns initialize0; original137 columns/bias copied. Metadata requires asset SHA, joint mapping, ground0, feature contract and separate offline artifact kind. State73, physical model, termination and normalizers unchanged.

Probe v1 passed; v2 adds physical-state reconstruction verification rather than replacing failed outputs. All4505 original fit states: initial predictions bit-identical to parent for both arms; save/load roundtrip exact; normalized-input reconstruction feature error3.2603741e-8m. Candidate feature weight gradient finite/nonzero; control gradient zero. No optimizer training or accuracy claim.

Inspection found ordinary policy loader lacked an explicit offline-kind guard (earlier shape/missing-field rejection was insufficient isolation). Added early rejection of offline_* before any model weights are loaded; exercised actual trainer.load against both saved experiment artifacts with a receiver lacking model attributes, obtaining the intended ValueError. Explicit offline geometry loader succeeds. Evidence: sole_geometry_contract_v2/audit.json; v1 retained. Next connect opt-in fitter and independent audits, then matched final-only training; no policy promotion.

---
### 2026-10-31T17:16:41.495685+00:00 | Geometry model fitter/audits smoke

Connected opt-in geometry features to existing fitter and explicit offline geometry artifact loader in all four trajectory/branch/native response auditors. Integrity auditor reconstructs source geometry buffers independently, checks immutable FK/input-normalization buffers, non-net freeze, exact metadata and zero XY input columns. Re-registered XY gradient mask on replaced first layer. Original numerical gates unchanged.

Both4-update arms pass independent smoke:768 endpoints/update, frozen original non-net tensors, finite raw/pose-integrated outputs, optimizer4 steps, zero XY columns. Control feature-column norm0; candidate0.0375853. Original no-feature fit rerun produces bit-identical weights to prior4-update artifact. Evidence: sole_geometry_fit_v1/smoke_integrity.json. Formal plan: two fresh original H8-parent arms,1000 updates each, all prior losses fixed including pose response1, same added parameters/FK budget, final-only original H1/H8/H32 and branch gates. No policy promotion.

---
### 2026-10-31T17:20:53.462501+00:00 | Fixed7 evidence contract while geometry trial runs

Verified actual fixed7 manifest SHAaed912a2f82867d7b76aeb35f19a58f9595f1d2ea9240794260255d6bfda7541 against original physics_v3_1000 baseline results. Fixed motion IDs and source families are disjoint from all332 training motions. Frozen individual motion NPZ hashes, original HumanML3D caption hashes, current quality flags and baseline original termination reasons. Existing required windows seconds:00324512.04,01258412.12,00336313.28,00831220,00504810.36,00451820,01019520. Last three long-reference IDs008312/004518/010195 have full retargeted durations22.44/24.94/30.58s: passing capped20s is not full-reference completion. Preserve original gates/windows and report any full-reference supplementary evaluation separately. Five clips have strict retarget quality false under inherited near-limit-advisory acceptance; no new relaxation or deletion.

First inventory execution failed caption path lookup before writing; corrected root. First successful inventory retained; v2 clarifies field name from full_source to full_reference (retiming means original HumanML3D source duration differs). Authoritative evidence: fixed7_evidence_contract_v2_20261101.json. Geometry training PID699260 confirmed live, no restart; data disk34GB free. Fixed7 policy success and >=3 seeds remain unachieved.

---
### 2026-10-31T17:23:33.918152+00:00 | Geometry control audit device reconstruction fix

Control completed1000, then independent integrity audit failed exact net.0.input_std equality. Original final checkpoint retained, candidate not yet started. Independent probe: CPU versus CUDA sqrt of frozen source variance differs by max1.1920929e-7; saved training buffer equals CUDA reconstruction exactly and differs from CPU. Training itself checked immutable buffers against its initial CUDA snapshot exactly. Fix auditor to construct expected source buffers on original CUDA device, then compare CPU copies exactly; do not relax equality or change weights. Failure log and std_device_initialization_probe.json retained. Resume script skips existing final control fit, audits it with v2 log, then starts candidate and unchanged gate chain; no retraining or checkpoint selection.

---
### 2026-10-31T17:26:12.855137+00:00 | Final gate summary provenance

Candidate training PID700057 and resume runner PID699909 verified live. Added read-only final gate summarizer binding final checkpoint hashes to independent integrity/branch audit evidence; reports original-parent, matched-control and held-branch failures separately, never policy promotion or final goal completion. Executed on completed rejected root_pose_response_fit_v1: original3/18, matched3/18, branch5/9 reproduced. Queued geometry summary against actual live runner PID, without restarting training or selecting intermediate artifacts. Evidence: root_pose_response_fit_v1/gate_summary.json; geometry summary pending actual completion.

---
### 2026-10-31T17:31:33.520457+00:00 | Geometry feature final review: long-horizon gain, still rejected

Both fixed1000 arms complete, independent integrity and full prescribed audits complete. Candidate SHA25687430b0009be5c8e4e8ea868f49e90f3df6bafc012aa7a3beea4d5f482598998. Matched trajectory18/18 nonregression checks pass. H32 root position p95 median63.4666->26.7601mm; orientation0.279122->0.170158rad; root linear RMSE0.0934784->0.0322859m/s; root angular RMSE0.423317->0.183823rad/s. Initial-parent gate still2/18 fail (H1 root angular, H8 orientation); branch6/9 fail (root angular/joint position/joint velocity absolute and all three required derivative improvements). Not accepted or promoted.

Native pose-response variation now greater, but not reliably correct: validation vertical-input angular response condition std control0.00004324 -> candidate0.00077601rad/s vs native0.00110959. Full response RMSE worsens0.00328666->0.00606485rad/s. Representation expresses more variation, but value/Jacobian constraints remain insufficiently consistent; not proof of a unique root cause. Evidence: sole_geometry_fit_v1/gate_summary.json, review.json, response_variation.json. Next fit-only loss/gradient conflict audit before another optimizer change. All checkpoints/failures retained; independent motion, >=3 seeds and videos still outstanding.

---
### 2026-10-31T17:35:07.787298+00:00 | Fit-only gradient diagnosis

Read-only audit across four fixed fit batches per final model completed, weights unchanged. Candidate median projected gradient norms: primary base0.0140142, velocity value0.0946602, action response0.0410929, root velocity response0.00572355, root pose response0.000297364. Root-pose/velocity-value gradient ratio~1/318; new feature-column ratio~1/31. Pose versus velocity-value cosine across four batches[-0.0141,0.0450], not evidence of universal strong conflict. Clip factors all1. No validation access or optimizer.

Evidence: sole_geometry_gradient_audit_v1/audit.json and provenance_and_summary.json (exact script/model hashes). Next preregistered measurement on fresh original H8 parent with geometry: all3249 fit H8 windows +1200 H1 states primary gradient, all48 fit pose conditions response gradient; fixed coefficient equals projected full-fit primary/pose gradient norm ratio, no coefficient search or validation selection. Only measurement planned, not yet executed; formal coefficient1 versus measured coefficient remains pending. Plan: sole_geometry_pose_gradient_scale_v1/plan.json. No policy promotion or full-goal acceptance.

---
### 2026-10-31T17:40:16.514827+00:00 | Full-fit measured pose coefficient and smoke

Predeclared source-parent measurement complete: all3249 fit H8 windows,1200 H1 states,48 fit root-pose conditions. Primary projected gradient norm0.00672701103, pose0.00124540184; coefficient5.401478318597007. No optimizer or validation, weights unchanged. Initial function error1.359731e-7. Prior final-model batch ratio318 is not used. Saved measurement and provenance bind source/plan/script and fixed metadata contracts; final learned geometry weights never used as measurement weights.

Two4-update coefficient1 vs measured smoke arms pass768 endpoints, frozen non-net, XY zero, optimizer4 steps and exact source bindings. Coefficient1 weights exactly match old geometry smoke candidate. Deliberately doubled coefficient rejected before optimizer/checkpoint creation. Evidence: sole_geometry_pose_balance_fit_v1/smoke_integrity.json and tampered_scale/candidate.fit.log. Generic geometry plan variable label clarified before formal runs (metadata only). Formal: same fresh588 parent, geometry enabled both,1000 each, original gates unchanged, no promotion.

---
### 2026-10-31T17:47:24.022454+00:00 | Full-budget coefficient1 repeat verification

Balance-trial control completed1000 and original integrity/trajectory audits; measured-coefficient candidate PID701469 confirmed live with coefficient5.401478318597007 and immutable measurement file argument. Compared new coefficient1 control against prior geometry candidate (also coefficient1): all18 key protocol fields and fit file hashes match; every world_model tensor is bit-identical, and all18 reported trajectory aggregate metrics equal exactly. This is a same-seed repeat, not three independent seeds. Evidence: sole_geometry_pose_balance_fit_v1/control_repeat_review.json and control_repeat_gate.json. No gate replacement, weight promotion or intermediate checkpoint selection.

---
### 2026-10-31T17:53:32.714680+00:00 | Fixed pose-gradient scale rejected

Both1000 arms and all original audits complete. Measured coefficient5.401478318597007 candidate SHA2563d22a5f52dd5f559ba1b2f73f21eeffa53fee2c6f83e45b8fa6c07994d6f47a0. Gate failures: initial2/18, matched3/18, branch5/9. H32 orientation0.170158->0.205384rad (+20.70%); root position26.7601->26.5709mm and root linear RMSE0.0322859->0.0275715m/s improve, but do not override failures.

Native pose-response validation median root-linear response RMSE0.000262560->0.000241973m/s while angular0.00279882->0.00336572rad/s worsens. Vertical-input angular response RMSE0.00606485->0.00805557rad/s. Scalar balancing is not a demonstrated remedy; no global coefficient search. Next fit-only per-output-group normalization and gradient audit, without assuming causality or changing gates. Evidence: sole_geometry_pose_balance_fit_v1/gate_summary.json, review.json, response_variation.json and original native response audits. All checkpoints/failures retained; geometry coefficient1 remains a rejected diagnostic comparator, not promoted policy. Final fixed7/3-seed/video requirements incomplete.

---
### 2026-10-31T17:58:38.774143+00:00 | Pose response groups and explicit corner velocity

All48 fit conditions split into half-weighted linear/angular losses, reproducing original six-output mean. Initial-parent group gradient norms0.00123942/0.00016503; trained geometry coefficient1 0.00007805/0.00028434; failed coefficient5.4 0.00006836/0.00037502. Linear/angular cosine -0.0303/0.0245/0.0512. Thus no evidence angular gradients are uniformly ignored throughout fitting; no general strong group conflict. Source normalization std linear0.260-0.377m/s, angular1.901-2.992rad/s, but these scales alone do not prove a defect. Evidence: sole_geometry_pose_group_audit_v1/audit.json. No optimizer or validation.

Implemented separate differentiable sole corner vertical velocity using world root link linear/angular velocity and joint velocities. Native CPU geometric-point velocities checked against differentiable FK for all480 pose branch states (max4.44451e-9m/s) and480 velocity branch states (4.93319e-9m/s), both below fixed1e-5m/s. Finite gradients and16-corner sum derivative w.r.t root vertical velocity=16 checked. No force/loaded-contact inference, dynamics stepping or policy change. Collection/report hashes bound in sole_corner_velocity_audit_summary_v1.json; original reports retained; generic branch wording clarified in script without arithmetic change.

Next predeclare height-only versus height+corner approach-velocity features, equal parameters and compute; feature velocity scale derived from existing height1mm divided by policy dt0.02s =0.05m/s. First verify model initial/serialization compatibility and fit coverage, then smoke/final-only matched fit. Hypothesis only; no promise of better dynamics/control, no coefficient search or gate changes. Final fixed7/3-seed/video acceptance remains incomplete.

---
### 2026-10-31T18:02:18.221050+00:00 | Velocity geometry contract and legacy regression

Added explicitly versioned height+penetration+vertical-velocity geometry contract,24 feature inputs. Both arms retain16 height/penetration features and compute the same FK with velocities; only the8 bottom corner velocity columns are enabled0/1. Velocity scale0.05m/s equals height0.001m / policy0.02s. Additional columns initialize zero, original137 columns and bias copied. State73, physics, termination and original statistics unchanged. Legacy height-only contract loader remains supported.

All4505 original fit states, each new arm: exact initial parent predictions and exact serialization roundtrip; height reconstruction max3.2603741e-8m, candidate velocity reconstruction1.7881393e-7m/s. New velocity column gradients zero in control and finite/nonzero in candidate. Both rejected by ordinary policy loader before weights. Legacy height-contract repeat also passes all4505 and exact initial/roundtrip checks.

Evidence: sole_geometry_velocity_contract_v1/audit.json, initial artifacts and verified_probe.json (explicit velocity-column semantics and raw/legacy report hashes); sole_geometry_contract_legacy_regression_v1/audit.json. No new training yet. Next fit-coverage check then opt-in fitter smoke and same-budget1000 final-only pair, retaining original physics and gates; no promotion or final goal completion.

---
### 2026-10-31T18:09:58.925477+00:00 | Velocity feature coverage, canonical XY and fitter smoke

Physical feature coverage uses fixed height1mm/velocity0.05m/s coordinates only. Fit3305 trajectory+1200 action+240 pose+240 velocity states; validation240 pose+240 velocity+4470 trajectory queries. Native pose-fit to pose-val nearest feature RMS p95~0.02587, velocity-fit to velocity-val~0.02593; trajectory-fit to trajectory-val p95~0.12677/max4.26812. Retain tails; these descriptive distances do not establish full coverage or tune normalization/samples. Evidence: sole_velocity_feature_coverage_v1.json.

Native dynamic trajectory fit velocity check all3305 states passed max2.6166811e-7m/s <= unchanged1e-5. To preserve exact horizontal invariance of velocity features despite floating point relative-position subtraction, compute only geometry feature FK at root XY0; actual physical states and position outputs unchanged. New contract probe v2 all4505 states passes exact feature translation invariance, exact initial predictions/roundtrip, max height reconstruction3.2603741e-8m and velocity1.9073486e-7m/s. Initial artifact default XY0 later recorded explicitly in fitter metadata; no arithmetic change.

Two4-update velocity input0/1 smoke arms pass independent source/freeze/optimizer/loader checks (768 endpoints). Control velocity-column norm0, candidate0.03132348; initial prediction errors0. Legacy height4-step training weights exactly match prior. Evidence: sole_velocity_feature_fit_v1/smoke_integrity.json and sole_geometry_velocity_contract_v2/verified_probe.json. Formal trial fresh588 parent, height features both on, root pose coefficient1, all other losses/data/RNG fixed,1000 final-only each, original gates unchanged. No policy promotion or full goal completion.

---
### 2026-10-31T18:18:27.198098+00:00 | Velocity-feature full-budget control complete

Control completed1000, original integrity passed (SHA256722656b51fa2559c9a8bdadd48865d6b52d0a79313fdcb0f408dc9ccf4a67ceb), full trajectory audit complete. Compared velocity-disabled new24-feature control against prior16-feature height-only final: all18 original nonregression checks pass; H32 position26.7601055->26.7606443mm (small numeric variation, not claimed bit-identical model). Evidence: sole_velocity_feature_fit_v1/legacy_height_control_gate.json. Current matched control remains this trial control, no baseline replacement. Velocity-enabled candidate PID703942 verified live under original runner, no restart. Final candidate/native response and fixed7/multiseed/video requirements remain pending; no promotion.

---
### 2026-10-31T18:31:11.720774+00:00 | Velocity feature rejected; dynamic replay next

Both1000 arms and all prescribed audits complete. Candidate SHA256334702937f36b382691800852ca78ff095b4366943bcecc2429ca4b57069df33. Initial-parent2/18 fail, matched2/18 fail, branch4/9 fail. H32 root position26.76064->20.94303mm and root linear RMSE0.032278->0.022652m/s improve; orientation0.170146->0.186308rad worsens9.50%, H8 angular RMSE +5.27% violates unchanged5% limit. Reject, no promotion.

Read-only residual decomposition complete across all archived fit/temporal-validation states, weights unchanged. Candidate root angular common-bias fraction branch fit80.89%/validation77.61%, but trajectory fit10.32%/validation5.35%; constant offset cannot explain most dynamic error. Candidate trajectory aggregate angular RMSE0.03881fit/0.11771validation vs original parent0.02668/0.06287. These are all-frame aggregates, distinct from gate median-per-trace numbers. Input coverage reports original137 normalized features, not added geometry; scope clarified in script. Evidence: sole_velocity_residual_bias_audit_v1/audit.json.

Next preregistered native dynamic replay, not another feature/weight search: all16 saved trajectories in original order/all7775 steps warm; cold state reconstruction at fixed first20/every50/final indices, no cherry picking. State73/context34 matching and actual native next-state errors, source/current physics equivalence and original terminations checked. Plan dynamic_state_replay_v1/plan.json; not yet executed. Full fixed7/3-seed/video objective remains incomplete.

---
### 2026-10-31T18:35:29.929498+00:00 | Native dynamic replay pipeline smoke

Implemented independent original one-environment replay with archived actions, context checks, source mappings/asset/physics metadata, all discrepancies and original terminations recorded. Warm preserves chronological trajectory history. Cold resets environment, restores73 state through world root link and joint writers, checks34 context, clears listed available solver/force/control/activation transient buffers, restores episode step count, then steps saved action; reference phase remains static frame0. This tests this explicit reset protocol, not universal hidden-state absence.

Warm smoke2 traces x3 steps and cold2 traces x(0/middle/final) completed within original1e-5 native replay tolerance; cold max input write error2.3841858e-7, next-state max1.9641325e-6. Candidate model residuals are not used in this test. Evidence: dynamic_state_replay_v1/smoke_integrity.json, per-phase raw arrays/results. Next full original-order16 trajectories/all7775 warm steps, plus predeclared first20/every50/final cold cases, no sample selection or original data changes. No model/policy fitting or promotion.

---
### 2026-10-31T18:47:09.284751+00:00 | Full native replay independent audit

All16 warm trajectories/all7775 transitions and fixed479 cold cases completed; archive/source hashes, original samples, exact cold index union, context, cold state restoration and warm step continuity independently verified. Original1e-5 precision checks fail: warm281 frames (max full-state0.07511997, cumulative input drift present), cold6/479 (max2.9655173e-5). No tolerance changes or sample deletion; universal Markov sufficiency not certified.

On identical restored cold validation states/actions/context, candidate model root angular RMSE0.12457649rad/s vs native-to-archive0.00000149985rad/s; original parent model0.10264966. Fit cohort candidate0.04910750 vs native0.00000091421. This bounds replay discrepancy on the fixed subset and does not explain most model error there, but is not full-motion/control acceptance. Evidence: dynamic_state_replay_v1/independent_audit/audit.json.

Fit accounting:3305 trajectory states include2000 zero/1305 dynamic;3249 H8 windows include1986 zero. Direct H1 pool currently1200 static-warmup action branches only. Next preregister H1 distribution0 vs half dynamic/half existing branches, same64 budget, independent seed47 for dynamic sampling, no validation or new features. Fresh588 parent/height-only geometry, all prior losses1 and original gates; plan dynamic_h1_mixture_fit_v1/plan.json, not implemented or trained. No promotion; fixed7/multiseed/video goal still incomplete.


---
### 2026-10-31T18:54:03.940122+00:00 Dynamic H1 mixture implementation

Fixed matched comparison: both arms start source588 with height-only geometry and prior loss coefficients1. H1 control keeps64 action branches; candidate uses32 branches plus32 dynamic fit states from6 nonzero-amplitude files (1305 states). Independent seed47 preserves original sampling generators; total768 endpoints/update and1000 final updates unchanged. Four-update smoke audit passed: legacy control weights bit-identical; candidate changes only8 dynamics network tensors; frozen tensors, source hashes and loaders checked. This is not held-out accuracy or policy acceptance. Original trajectory/branch gates and all failed native replay cases remain unchanged. Formal paired run is the next step.


---
### 2026-10-31T19:07:17.381103+00:00 Dynamic H1 mixture rejected

Both final1000 arms passed integrity. Control tensors exactly reproduce prior height-only candidate. Candidate dbf20e87c7793d99c0cea908086e8f9e780bff3a10e35b3a2bd8582277d502fb fails3/18 original-parent trajectory checks,8/18 matched checks and7/9 branch checks. All-frame temporal-validation angular RMSE: source0.062875, control0.117372, candidate0.099946 rad/s. This aggregate improvement over control does not override original failed per-trace/multi-step gates or worsening branch angular RMSE0.012634 to0.017901. No policy integration/promotion. All checkpoints, native response audits and failures retained. Read-only coverage audit confirms unchanged model tensors. Auxiliary coverage queue initially failed due missing shell variable; original script/log retained and fixed queue completed without touching training. RunningNorm.normalize source is affine plus frozen passthrough, with no clipping; clipping is not the explanation, while representation/conditioning and loss tradeoffs remain unresolved. Standing, fixed7 dynamic control,3 seeds and final videos remain outstanding.


---
### 2026-10-31T19:09:43.521172+00:00 Dynamic H1 gradient diagnosis

Read-only8 fixed fit batches completed; all world-model tensors unchanged. Control branch-vs-dynamic H1 cosine0.347 to0.792, candidate0.595 to0.867: no general strong negative conflict in these probes. Action-response vs dynamic angular cosine control -0.0393 to0.0010, candidate -0.0338 to-0.0045, near orthogonal rather than a strong universal opposition. Velocity-value vs dynamic angular positive in all8. Pose-response sign mixed. These local observations do not prove causality or justify automatic gradient projection/scalar scanning. Preserve failed dynamic mixture and original gates; representation/conditioning and finite training-objective tradeoffs remain open. Reproducible audit script uses actual H1 mixture for each model, and excludes three diagnostic probe gradients from primary total/clip statistics.


---
### 2026-10-31T19:11:57.014621+00:00 Full amplitude stratification

All16 original traces,4 fixed forcing amplitudes and H1/H8/H32 retained; exact window-weighted velocity RMSE derived from existing finite frozen predictions, no new fitting/gates. Validation H1 angular RMSE by amplitude0/.02/.05/.10: source .006922/.036103/.125292/.244075; control .010692/.056537/.192781/.503859; mixture .016286/.051811/.168686/.418512 rad/s. Mixture improves over control at all nonzero amplitudes yet remains worse than source at each; zero amplitude worsens. This is not only a rare largest-amplitude failure. Existing calibration is same synthetic static reference with action perturbations; it does not establish walking-reference/contact coverage. Verified training manifest includes000075, KIT/3/walking_slow02 source,370frames, strict retarget quality passed. Next data-extension design must keep332 training sources isolated from fixed7 families, original terminations and old calibration gates; no claim that a PD reference replay is learned walking or physically feasible without measurement.


---
### 2026-10-31T19:16:24.564047+00:00 Native motion-reference PD collection

Predeclared first8 distinct source families in ascending existing training-category walking IDs:000041/000075/000083/000112/000132/000146/000152/000203, alternating4fit/4validation, no quality or result filtering; labels can include running/mixed movements. Families/IDs disjoint from fixed7, but these are previously used training sources, not untouched policy holdout. Four phases0/.25/.5/.75, at most100 steps or remaining reference frames, original first termination retained; deterministic reset/events off, no autoreset, original v3 caps. Reference joint target inverted through actual action mapping; no learned policy. First2 clips frame0 each3step smoke and independent audit passed: exact reference timing/joint mapping, continuity/finite/caps/terminal checks, maximum inverse error5.96e-8rad; independent audit also binds manifests and exact first8 selection. Formal32 snippet collection next, no walking-success claim and no original calibration gate changes.


---
### 2026-10-31T19:21:05.001663+00:00 Native motion collection and frozen-model review

32 snippets/1525transitions collected and independently audited.30 early terminations retained; ee_body_pos25/anchor_pos10/anchor_ori1 may coincide. Fit828states716H8/337H32windows, validation697states585H8/211H32windows. Only2 snippets reach their short remaining-reference windows, neither is20seconds/fullclip success. Frozen source/height/mix validation H32 position p95 episode median0.8466/0.6985/1.3880m and angular RMSE episode median15.175/32.677/28.024rad/s (13/16 eligible snippets211windows, no padding). Existing static calibration is insufficient for these motions. First2 snippets native warm3steps each maxstate2.21e-6; cold first/middle/final6cases maxstate5.07e-7, statewrite2.38e-7; all within unchanged1e-5 native tolerance. Preregister full warm1525 and cold96 covering all32 snippets, correct original reference phases and world root velocity restoration. Keep all replay failures and checkpoints; no new policy fit or original gate changes.


---
### 2026-10-31T19:26:18.517865+00:00 Motion replay and stale command cache

Independent audit: warm164/1525 exceed1e-5 (max.0033547, cumulative changed inputs), all terminal flags match; cold96/96 next states within1e-5 (max2.861e-6), but2 terminal mismatches at000083_frame90 index37 andframe181 index51. reset_to_frame writes clip/time/physics but does not update_relative_body_poses; the termination cache can remain from reset. Isolated optional --refresh-command-cache cold replay updates it after restoring physical state: all96 remain within1e-5 (max2.384e-6), terminal mismatches2 to0. V1 outputs untouched; V2 links original warm files and stores separate cold outputs. This supports cache synchronization as explanation for these terminal differences, not general Markov/control success. Same-query validation native/archive angular RMSE5.337e-7 vs source/model1.3251, height2.1114, mixture2.1339rad/s; model error remains. Future motion collection must explicitly synchronize and record current-frame command cache before fitting; original files and all warm failures retained.


---
### 2026-10-31T19:28:27.800313+00:00 Cache-consistent motion collection

Optional refresh-command-cache collection requires explicit current_state_current_frame_v1 plan contract, refreshes relative-body targets after reset/forward, records every pre-step command_body_pos and tracked-body order/env origin. Independent audit verifies cached target Z against source frame body Z plus origin to1e-6, source family/file/frame/split exact first8 selection and fixed7 separation, fixed planned windows, mappings/finite/caps/continuity/terminal records. Legacy invocation remains supported and original v1 data retained. New2clip3step smoke passed including cache-Z audit. Full32 snippets use unchanged source/phases/actions/physics/thresholds in separate motion_pd_collection_cache_v2; no learned-policy or success claim.


---
### 2026-10-31T19:34:13.285789+00:00 Motion H8 mixture implementation

Correct window counts:828 fit states yield716 H8 and337 H32 windows; an earlier prose note confused337 with H8, actual preregistration/data always716. Same fresh588 source, height-only geometry and all prior coefficients1. Control H8 batch64 old windows; candidate32 old+32 audited motion fit windows, independent seed48 draws in both arms, original seed42/43/44/45/46 draws unchanged. H1 remains64 original branches and768 total prediction endpoints/update. Loader opens only16fit files from4families, binds source hashes, audit/cache contract, physical asset/contract, mapping and terminal/continuity checks. Four-update paired smoke passed: legacy control world_model tensors bit-identical, candidate changes only8 dynamics network tensors; original frozen normalizers/termination and geometry/XY invariants retained. Formal1000 both arms follows unchanged18 trajectory/9 branch gates and separate new-motion H1/H8/H32 with explicit eligible windows. No policy fitting or promotion.


---
### 2026-10-31T19:48:24.141469+00:00 Motion H8 mixture rejected despite motion gains

Both1000 final models passed integrity; control tensors exactly match old height model. Candidate080494ef844ecdb66f002bf7eb4bccb4d637a2b242838498ac1967eb32b807d4 failed2/18 source-parent,11/18 matched-static and6/9 held-branch checks. Keep all gates/checkpoints; no policy training or promotion. Same new source-isolated motion validation H32 has13/16 eligible episodes211windows: position episode-p95 median .698513 to .231442m, orientation2.66379 to .963352rad, angular RMSE episode median32.6768 to1.86964rad/s. H1 angular1.71332 to .504615; large motion improvement does not override static regressions or establish accurate physical control. Next preregistered read-only global yaw covariance probe uses all32 snippets first/middle/final96 states and0/90/180degree transforms; native symmetry must be measured separately before treating rotated archive endpoints as physical truth. No blind mixture-ratio scan. Fixed7/3seed/fullcontrol/video requirements remain incomplete.


---
### 2026-10-31T19:55:53.984898+00:00 Quarter-turn model/native covariance

All32 motion snippets first/middle/final96 states and0/90/180degree transforms measured. World positionXY/quaternion/worldlinear/angular velocity rotate; joint/action/context fixed. Independent NumPy quaternion/vector oracle and inverse transforms pass;3 model weights unchanged. Motion-H8 model validation48-state covariance angular RMSE .972535rad/s at90 and1.427057 at180. Native all288 queries:0 endpoint failures at original1e-5, rotational angular discrepancies~1e-6rad/s;4 input-write failures at original2e-6 (1at90,3at180,max2.861e-6), retained. Thus cannot call all native checks passed, but the measured model discrepancy greatly exceeds native numerical deviations on tested states. First90 run aborted on2.384e-6 write error; preserved and reran all96 in separate complete directory with failures recorded instead of aborting, no threshold relaxation/sample filtering. Initial independent audit tried bit-exact GPU/CPU transformed quaternion equality and failed at5.96e-8; corrected to pre-existing2e-7 transform oracle bound only for quaternion, other target components exact, native/state-write bounds unchanged; all failed logs retained. Source archives/model/native outputs hashed. Next isolated coordinate-frame canonicalization probe must address yaw and quaternion-sign roundtrip, gradients and physical old/new gates; not coefficient tuning, policy integration or acceptance.


---
### 2026-10-31T20:04:22.096014+00:00 Explicit yaw-frame prototype, not accuracy acceptance

Fixed144 states (all16static and32motion first/middle/final),3 historical models, no fitting. Canonicalize XY0/yaw0 with world linear/angular velocities rotated; normalize quaternion positive-w hemisphere, FP64 coordinate operations, FP32 network features, explicit heading singularity error. NumPy oracle, inverse(max2.384e-7), antipode, disabled path and all144 state/action gradients pass; minimum heading projection .263276. Source588 and height215 covariance pass1e-5. Historical motion080 fails FP32 quarterturn maximum1.502e-5/1.478e-5, retained; full probe NOT passed. Separate FP64 rotated-input attribution gives max4.44e-15 while keeping network featuresFP32, supporting rotated-observation rounding amplification rather than mathematical yaw inconsistency; does not make FP32 failure pass. FP32 regression after dtype support within1.907e-6. Initial function changes with unchanged weights: static-validation sampled angular RMSE source .322693 to .453327, height .371925 to .498624, motion .336509 to .434816; cannot directly deploy/relabel old checkpoints. Next preregister fresh588 coordinate0/1 fit, hold prior motionH8 mix1 and all budgets/gates fixed; first integrate strict loader/derivative audit semantics and smoke control equality. No artifact/policy loading or training yet; retain fixed7/3seed/video requirements.


---
### 2026-10-31T20:12:40.966859+00:00 Strict yaw artifact and paired smoke

New explicit offline_yaw_canonical_dynamics_not_policy_checkpoint kind requires matching coordinate/geometry/pose/plan metadata; missing or unknown coordinate fields cannot silently fall back to ordinary inference, including removal of plan declarations. Shared loader composes Pose(Yaw(raw)); geometry loader now uses it too. Value trajectories and all3 response losses use same canonical physical velocity function; responses use Yaw(raw) because Pose leaves velocities unchanged. Both fresh588 arms keep motionH8 mix1, old H164, all prior coefficients1 and768 endpoints. Primary initialization proof binds fresh588 passing source row only; historical080 FP32 failures remain visible, aggregate probe not relabeled passed. Four-update smoke passed: control exact old motionH8 candidate weights, only8 dynamics network tensors differ; frozen/XY/source/optimizer counts verified. Shared vs explicit manual predictions and response velocities agree,6 negative contract cases per arm and actual policy loader reject offline yaw artifacts. Formal1000 pair next, old18+9 gates plus new-motion and explicit144-state FP32 coordinate reports; no policy integration or promotion.


---
### 2026-10-31T20:32:25.762907+00:00 Yaw fit rejected; true-velocity integration oracle

Yaw candidate24121c289f808bc48afb7e97653fb7b86f82672aae0073a85e48e69cf131c91d: both1000 integrity pass, original-parent3/18, matched-static5/18 and held-branch7/9 fail; FP32 covariance max1.955e-5/1.907e-5 also fails1e-5. Motion validation H32 eligible13episodes211windows root p95 episode median .231442 to .156239m, angularRMSE1.86964 to1.59986, but H8 angular .864305 to .934006 worsens. No promotion; all failures retained.

Fixed original alpha, no refit: all16static+32motion H1/H8/H32 and40branch H1 true-velocity oracle completed, constant-velocity identity pass and exact velocity preservation. Motion validation H32 oracle root position2.2235mm, orientation.0326468rad, joint .0135172rad; learned candidate156.239mm/.937600rad, so tested root error is predominantly beyond fixed-integration approximation. Oracle is not deployable or a global lower bound; biased velocities can compensate pose error. Fit-motion learned H8 root6.354mm vs H32 90.799mm, oracle fit H32 1.912mm. Predeclare same512 rollout endpoint comparison:64H8 vs32H8+8H32, retain short motion episodes in H8, auxiliary32 windows only where long enough, no padding or sample replacement; H164 and response192 unchanged(total768). Independent49/50 samplers, all prior mechanism/budgets/gates fixed, fresh588; implementation next.


---
### 2026-10-31T20:39:45.979634+00:00 Same-budget mixed-horizon implementation

Fresh588 controls yaw1/motionmix1 and all prior mechanisms. Control64H8, candidate32H8(16static/16motion)+8H32(4static/4motion),512 rollout points; endpoint-weighted .5H8+.5H32 value/velocity losses, originalH164 and response192 retained(total768), independent49/50 generators consumed both arms, original samplers unchanged. Static H32 pool3057, motion H32 pool337from14 eligible episodes; all16motion files/828states retained in provenance and H8 pool716windows, no padding/reset crossing or sample replacement. DefaultH8 loader metadata unchanged. Four-update source/freeze/control/negative-loader audit passed, then add actual raw-forward point counting and repeat separately: everyupdate768,4total3072 botharms, golden control tensors exactly equal prior yaw smoke candidate; originalclip kept despite large candidate gradients. Formal1000 must record768000points, old18+9 and full motion/FP32-coordinate diagnostics unchanged. No policy promotion; fixed7/3seed/videos still incomplete.

---
### 2026-11-01T02:17:42.303999+00:00 mixed H8/H32 final rejection

Both arms completed 1000 updates and integrity audits. Candidate 49ceaa9fe23f919b3cc5315fb33ff238521c8b2ecdebf050bc74e63cfe144411. Failed original-parent 9/18, matched-control 13/18, held-branch 9/9. Motion validation H32 episode-median position p95 0.156239 -> 0.201394 m, orientation 0.937600 -> 1.122318 rad, angular velocity RMSE 1.599855 -> 2.034884 rad/s. Fit H32 position 0.090799 -> 0.090419 m but orientation 0.360560 -> 0.440836 rad: longer training horizon did not solve fit or generalization. Candidate FP32 90-degree covariance max 1.45435e-5 still fails unchanged 1e-5. Evidence: data-disk mixed_horizon_fit_v1/final_gate_summary.json, motion_pair_medians.json and review.json. Preserve both checkpoints; no promotion. Next: fit-only gradient decomposition and recurrence-error audit before another training change. Physical fixed7 and >=3-seed acceptance remains incomplete.

---
### 2026-11-01T02:20:38.376182+00:00 fit-only recurrence audit

Script audit_supermimic_recurrence_fit.py checked all337 audited fit H32 windows per final arm, no validation arrays or optimization. Finite outputs and bitwise unchanged model tensors verified. Control step32 teacher-forced/free angular RMSE 0.173815/0.683704 rad/s; candidate 0.511799/0.841557. Candidate local velocity accuracy deteriorated, not solely recurrence amplification. Four matched velocity-only gradient probes had positive H8/H32 cosine in both arms (control 0.480--0.955, candidate 0.269--0.940), so these probes do not support direct opposing gradients. These are not full-objective or universal conflict tests. Candidate recurrent linear velocity 0.162155 versus control 0.189099 m/s, while teacher-forced 0.094893 versus 0.022050: possible long-horizon/local-accuracy tradeoff, not causal proof. Evidence data-disk recurrence_fit_audit_v1/results/audit.json. Next inspect H1 static-only coverage versus actual-motion single-step supervision before preregistering a fixed-budget single-variable change. No promotion or physical acceptance.

---
### 2026-11-01T02:39:19.601420+00:00 motion H1 anchoring rejected

Both arms1000 updates/integrity passed; control bit-identical legacy yaw/H8. Candidate 2cc3a41f55a8d328254ef8925dcdd16c90ac7a518bd98d948b6e006b2d98cb65. Parent4/18, matched5/18, held-branch7/9 failed. Candidate fixed144 FP32 coordinate audit passed unchanged1e-5, control failed. Motion validation H1 position p95 3.087->1.620mm and angular RMSE0.492260->0.391513rad/s improve, H8 position31.637->27.924mm improves; H32 position156.239->175.821mm and angular RMSE1.599855->1.677116 worsen, orientation0.937600->0.797405rad improves. Fit H32 position90.799->71.064mm improves. No promotion; frozen fit-only recurrence audit next. Evidence motion_h1_anchor_fit_v1/final_gate_summary.json, motion_pair_medians.json, review.json. Original physical fixed7 and three-seed acceptance incomplete.

---
### 2026-11-01T02:41:23.952981+00:00 H1 recurrence diagnosis

Frozen all337 fit H32 windows, finite outputs, unchanged weights, no validation arrays. Step32 teacher-forced linear RMSE0.022050->0.007643m/s and angular0.173815->0.119840rad/s improve, but free recurrent linear0.189099->0.281642 and angular0.683704->0.973965 worsen. Local accuracy does not imply rollout stability. Four matched H8/H32 velocity gradient cosines remain positive (candidate0.404--0.898). Evidence motion_h1_recurrence_audit_v1/results/audit.json. Next isolate predicted root pose/velocity/joint feedback with frozen fit-only diagnostics before another training change. No promotion. Remote087b234 push was verified by ls-remote.

---
### 2026-11-01T02:44:01.248428+00:00 frozen feedback-group sensitivity

All337 fit H32 windows per arm,32steps,6 input modes; finite, weights unchanged, first-step substitutions agree<=1e-5. No validation arrays or optimizer. Each one-group truth transplant is queried on the SAME unmodified free rollout and is never fed back. Candidate step32 angular velocity RMSE0.973965rad/s becomes0.288811 with only root velocity truth,0.859229 with root pose truth,0.977561 with joint position truth,0.980682 with joint velocity truth. Control0.683705 becomes0.280708 with root velocity truth. Candidate linear RMSE0.281642 becomes0.147328 with pose truth or0.144770 with root velocity truth. Root feedback is implicated as a network sensitivity; hybrid inputs may violate physical consistency, so this is not causal physical proof. Evidence feedback_group_audit_v1/results/audit.json and review.json. Next compare native/learned root-velocity response at off-trajectory fit queries before choosing training changes.

---
### 2026-11-01T02:50:10.065598+00:00 native off-trajectory audit

All288 queries/576 cold restores completed; coverage/archive checks passed, original2e-6 write and1e-5 repeat/true replay failures all0; weights unchanged, no validation arrays. At index31 (14 fit snippets), model/native angular RMSE at true inputs control0.240589/candidate0.172720rad/s; at each own free predicted input0.459291/0.294983. Candidate is locally more accurate on these off-trajectory queries too. Native next versus original archived next at free inputs0.910871/0.834984, while model versus archived0.735717/0.857085; MSE cross terms -0.499355/-0.049619. Baseline apparent rollout advantage partly reflects error cancellation, not uniformly better physics. Own-model query states differ, so this is not matched-state universal superiority. Evidence offtrajectory_native_audit_v1/full/results.json, review.json, review_decomposition.json. Next preregister source-isolated native root-velocity branches near real motion states, addressing static-only response coverage. No gate changes, promotion or final physical acceptance.

---
### 2026-11-01T02:57:01.032428+00:00 motion root velocity branch acquisition

Predeclared all32 motion snippets,first/middle/final states,6root velocity axes,+/- old0.005m/s or0.02rad/s,levels1/2,two swapped sign assignments plus nominal12:480batches/5760endpoints. Four fit versus four validation source families unchanged. Full12-world collection failed original1e-5 paired repeat bound twice, both validation000203 (frame205 index17 level2 max1.28746e-5; frame308 index21 level1 max1.70469e-5), restored input differences5.22e-8/1.19e-7. Preserve full/integrity.json and repeat_failures.json; not approved for fitting. Initial smoke mapping shape12x30 vs1x30 failure retained, explicit replication fixed shape without numeric tolerance change. Serial same-world60-endpoint smoke passed; serial full acquisition running in tmux motion_root_velocity_serial. No sample deletion, threshold change, optimizer or promotion. Auditor additionally verifies original nominal next state/terminal replay and source contracts.

---
### 2026-11-01T03:05:24.391507+00:00 serial motion root branches passed

All480 batches/5760 endpoints serial acquisition passed original write/repeat/nominal-next/terminal checks with0 failures. Preserve failed12-world data; no threshold changes. Frozen source588/control997/candidate2cc evaluated on source-isolated fit/validation576conditions each. Validation angular absolute RMSE1.325505/0.884472/0.720522rad/s, but normalized root-response MSE0.110454/0.093330/0.097422: H1 improves absolute prediction but does not improve held motion response versus control. Fit response0.103742/0.089072/0.090585 shows same pattern. Evidence motion_root_velocity_branches_v1/serial_integrity.json and motion_root_response_models_v1/results_v2/audit.json. First evaluation incorrectly used geometry-only loader for ordinary source and was rejected; original log retained, fixed common explicit-contract loader path without relaxing type checks. Next preregister fixed-budget static/motion response mixture with all other inputs/weights unchanged; no promotion or physical acceptance.

---
### 2026-11-01T03:25:14.340045+00:00 motion root response mix rejected

Both1000-update arms/integrity passed; control bit-identical old H1 candidate with768000 measured predictions. Candidate 159d897bae33815d5b9e0bcdeb91aa5ab10fd2b85612cf77de8bce650d8eb8b8. Parent3/18, matched8/18, held branch7/9 fail; FP32 coordinate candidate fails/control passes. Motion root response normalized MSE fit0.090585->0.037774,validation0.097422->0.073375 improves24.7%, but absolute angular RMSE fit0.193998->0.435013,validation0.720522->0.875895rad/s worsens. Original motion validation H32 position p95 episode median175.821->139.961mm,orientation0.797405->0.570706rad improve,angular1.677116->1.837369rad/s worsens; H1 angular0.391513->0.603795 worsens too. Evidence motion_root_mix_fit_v1/final_gate_summary.json,motion_pair_medians.json,motion_root_response_validation/audit.json. Next fixed-fit gradient relation between response and absolute state supervision before new changes; no coefficient scan, gate changes or promotion. Physical fixed7/three-seed goal incomplete.

---
### 2026-11-01T03:29:43.276180+00:00 response-gradient and input-scale evidence

Frozen4 fixed fit batches per final arm, no optimizer or validation arrays. Candidate motion-response/base gradient cosine -0.0263..0.0866 and response/velocity -0.0959..0.1140, mixed near-orthogonal; all clip factors1. No universal strong conflict evidence or PCGrad justification. Evidence motion_root_gradient_audit_v1/results/audit.json. Actual137 first-layer input after shared yaw canonicalization: static trajectory joint-position abs p95/max1.189/1.844 and action1.145/1.714; fit motions joint15.751/24.297 and action11.517/13.227. Root angular p951.172 and linear1.482. Inputs finite, weights unchanged. Evidence fit_input_scale_audit_v1/results/audit.json. Ranges alone do not prove ill-conditioning. Next preregister training-only joint/action affine reparameterization using fit-only frozen stats, initial-function preservation and export to original inference coordinates; normalizers/losses/budget/gates unchanged. No coefficient scan or model promotion.

---
### 2026-11-01T03:33:48.630561+00:00 training-only input-scale contract

Fixed60 joint/action dimensions13:43,73:103. Mean0, scale=max1,sqrt(0.5mean static normalized x^2+0.5mean motion normalized x^2), fit3305/828 populations equally weighted; remaining93 dims identity. Geometry computed from unchanged original input BEFORE scaling concatenated153 features. Wtrain=Wraw*scale,xtrain=xraw/scale; exportWraw=Wtrain/scale. Frozen original normalizers and losses unchanged. Full6437 fit queries passed initial max1.90735e-6 and genuinely perturbed export max2.20537e-6 at unchanged1e-5; disabled bit-exact, finite state/action gradients, no optimizer/validation. Initial indexed add_ perturbation only changed a temporary; initial result retained and explicitly limited. Fixed explicit assignment+change assertion and reran entire proof in training_input_scale_contract_v1/results_v2/audit.json. Next integrate optional training coordinates, smoke control regression and optimizer/export contract checks before formal training; no efficacy or policy acceptance claim.

---
### 2026-11-01T03:37:03.114905+00:00 training input-scale paired fit launch

Optional training-only scale0/1 integrated after unchanged geometry features; original inference world weights exported, separate training_coordinate_optimizer/first_linear with explicit non-raw-resume contract. Fresh588,seed42,1000 updates,Adam1e-4,all original coefficients1,root-response mix1 and actual768 endpoint budget unchanged. Both4-update smoke passed legacy-control bit-exact regression, measured3072 endpoints, frozen/source/fit-isolation checks, independent training-first-layer reconstruction and original-coordinate prediction agreement<=1e-5, offline policy rejection and6invalid contracts. Full6437-query initial/export checks in fitter bind frozen fit-statistics proof. Evidence training_input_scale_fit_v1/smoke_integrity.json. Start formal tmux training_input_scale_fit; original gates and supplementary motion-root evaluation, final-only checkpoint, no promotion.

---
### 2026-11-01T03:54:02.905746+00:00 training input-scale candidate rejected

Both1000-update arms/integrity passed, control bit-identical159d prior candidate; candidate 8ecb393830a0b6c1a922fc02ec5912628e2bdc1dfd9d114c5ef88e076e1a741a. Full6437-query original-coordinate export max2.04891e-6 passes1e-5. Parent4/18, matched3/18, held-branch5/9 fail; bothFP32 coordinate audits fail. Motion validation H32 position p95 episode-median139.961->164.799mm worsens,orientation0.570706->0.558676rad and angular1.837369->1.621077rad/s improve. Motion root response held MSE0.073375->0.074886 worsens slightly. No comprehensive gain, no scale scan/promotion. Evidence training_input_scale_fit_v1/final_gate_summary.json,motion_pair_medians.json,motion_root_response_validation/audit.json. Next audit reference joint-angle finite differences/velocity against unchangedv3 physical speed caps; check existing evidence first, no reference/cap/gate/sample changes. Physical fixed7/three-seed acceptance incomplete.

---
### 2026-11-01T03:57:13.966683+00:00 reference speed cap necessary-condition audit

All339 original motions covered with hash/30joint/fps50/finite checks. Fixed7 stored and adjacent-angle speeds have0violations,maxratio0.990017; not a dynamic feasibility certificate. Training332:stored velocity1motion009957,1frame,maxratio1.000949; adjacent-position rates10motions005297,005438,005845,007210,009957,010123,011770,012313,013591,014594,1--2 intervals each,maxratio1.239744 at011770 right ankle yaw. None is among original8 motion-PD source IDs, so these isolated violations cannot explain the widespread diagnostic terminations. No files, timing, caps or sample membership changed. Evidence reference_speed_cap_audit_v1/results/audit.json. Next inspect existing nominal/contact evidence and verify sole support/COM rather than another blind fit change. Final physical acceptance incomplete.

---
### 2026-11-01T04:01:13.279951+00:00 loaded foot support versus COM

Existing pose/COM/contact records reviewed before new analysis. All780 archived contact samples aligned to same env.step next_state, MuJoCo COM and loaded-foot XY hull with original nominal0.01N threshold; hull square/triangle/degenerate self-tests pass. Nominal PD400samples all2D hull and0COM-outside; after2s360 samples margin median8.75191mm,min8.74864mm. This does not support nominal posture necessarily lacking face support or COM containment. Perturbation0.02:289samples14outside;0.05:56samples21outside/2degenerate;0.1:35samples8outside/8degenerate. Durations end at original terminations, no20s extrapolation. Foot-only last-solver contact data versus post-stepCOM can have substep timing offset; no momentum/friction cone or causal-fall proof. Evidence loaded_support_com_audit_v1/results/audit.json; existing bounded/arm10 geometry and independent forcing confirmation reviewed. Next inspect real-physics PPO and predeclared low-arm posture evidence for multiseed closed-loop standing comparison, while fixed7 full-motion acceptance remains unchanged/incomplete. No failed world-model promotion.

---
### 2026-11-01T04:12:20.036412+00:00 real PPO seed support and repeat regression

Preregis real_ppo_multiseed_standing_v1: same static posture/parent92a3be, baselinecoef0 vs smooth-distancecoef2, trainingseeds42/43/44,256env*32*1000 each. Existing42 artifacts verified at original budgets; new43/44 runs not launched. Added explicit--seed default42 and matching artifact-audit expected seed, all other defaults unchanged. Four original smoke runs plus unchanged legacy42 repeat passed training integrity; however default42 runtime bitwise actor regression fails (max0.002249), and old-vs-old repeat also fails(max0.002867), with identical plan/final torch RNG. Default42 source AST operation equivalence passes after only seed default substitution/CLI validation removal. Preserve strict regression as FAILED; AST does not relabel GPU test. Initial reuse verifier optional historical sole fields KeyError retained, corrected to existing original audit defaults0/squared, artifacts unchanged. Fixed process PYTHONHASHSEED0 old/old/new repeat running in tmux real_ppo_hash0_smoke; no long training, physics changes, gate relaxation or promotion. Evidence seed_equivalence_v2.json and original smoke_audit.log.

---
### 2026-11-01T04:16:23.863629+00:00 multiseed real-physics standing evidence collection

Fixed hash0 old-old actor difference0.003253 and old-new0.002730,identical plan/torch RNG:hash fixation did not fix native replay. Preserve runtime_exact_regression_passed=false; AST/default operation equivalence and all5smoke/2reused42 artifact audits passed. Do not relabel bitwise test or policy acceptance. Proceed with preregistered independent real-physics research runs43/44 baseline0 versus smooth-distance2,same256*32*1000,source92a3be,all original physics/noise/reset/termination/normalizers unchanged,WMfrozen0calls. This collects statistical control evidence despite documented native replay limitation; does not resolve/waive strict pairing or promote a policy. Reuse immutable verified42 artifacts only. Sequential tmux real_ppo_multiseed_standing,100-update checkpoints,~6.8h newtraining plus nominalfront/side video and fixed512factorial evaluations each. Original strict1e-5factorial pairing failures retained independently; later runs can collect evidence but no causal/acceptance pass inferred. Data/cache all ondata disk,10GiB reserve; incomplete runs stop for explicit checkpoint recovery without overwrite/restart. Full fixed7/3seed finalgoal incomplete.


---
## 2026-11-01T12:05Z: fixed three-seed standing comparison and dynamic preflight

All six prescribed final-update policies (baseline/smooth sole distance, seeds 42/43/44) completed training and nominal/factorial evaluation. The authoritative data-disk aggregate is `longterm_diagnostics/real_ppo_multiseed_standing_v1/multiseed_summary.json` under the existing SuperMimic artifact root. All nominal trained episodes reached 20 seconds; this is static T-arm standing, not walking. Front/side media decoded fully; sampled visual review does not replace full temporal review.

Candidate minus baseline completed-20-second counts, profiles ordered 000,001,010,011,100,101,110,111 (reset/sensor/action toggles), 32 worlds per profile:

| Training seed | Count differences |
|---|---|
|42|0,+4,0,+2,+2,+9,+5,+3|
|43|0,+2,0,-3,+5,+4,0,+1|
|44|0,-8,0,-1,-3,+4,+1,-1|

Nominal joint-error deltas were -0.031524, +0.020989, +0.034654 rad respectively. Do not average away subgroup regressions or promote the candidate. Final seed44 candidate factorial archive invariants passed, but strict numeric pairing failed (max PD sensor-toggle state discrepancy 0.000102996826171875 versus unchanged 1e-5 bound). Its results SHA256 is `7ab863ba4ffbe041829f954fb5fc003df3d7f1530a333c399e0a115c5f0995b1`. Earlier numerical failures remain failures.

After the standing runner released its lock, tmux `dynamic_ppo_preflight` ran the unchanged real-PPO entrypoint on the original 332-motion training manifest: seed42, 8 environments, 4 updates, 1024 real transitions, original parent with zeroed action readout, no static sole penalty. Serialized artifact audit passed; final SHA256 `0704ebe4f44085935f7f4c0b93ffce6b68f07f512fcac5df23773bcd6e89b95b`. World model and source normalizers remained frozen. 49 episodes ended, including 13 timeout events; mean duration of ended episodes was 0.386122s. This tiny run establishes executable updates, not a trained walking policy or a stable-performance estimate.

Artifacts: `longterm_diagnostics/dynamic_real_ppo_preflight_v1/{plan.json,queue.sh,sampling_contract_audit.json,artifact_integrity.json,smoke/metrics.jsonl}`. Source review corrected a prior note: manifest weights are not consumed by this entrypoint. Return-based reset sampling starts uniformly across 3238 one-second bins, then uses uniform/negative-return-softmax/past mixture 0.5/0.4/0.1. Listing 332 files does not prove actual coverage in this smoke.

Next: instrument actual dynamic reset/reference coverage, original termination reasons and normalized actor/critic/goal ranges without changing policy or simulator contracts; compare against nominal/static ranges before choosing a staged weight-shift/step intervention. Review retargeted foot lift and loaded sole contact on those actual samples. Preserve all failed runs, all checkpoints, and original fixed7 windows/thresholds. Full SuperMimic world-model acceptance, dynamic fixed7 success and >=3 training-seed validation remain unmet. No policy promotion.


---
## 2026-11-01T12:10Z: observed dynamic inputs and original termination reasons

`probe_supermimic_real_ppo_inputs.py` wraps the existing collector/environment for instrumentation; it does not change rewards, original termination gates or sampler draws. Normalizer forward calls are read-only, with CPU/CUDA RNG equality assertions around observations. Additional synchronization can change GPU replay; no bitwise physical equivalence is claimed. Separate static and dynamic runs each completed 8 environments x 32 steps x 4 updates = 1024 transitions; both independent serialized artifact audits passed. All artifacts remain under data-disk `longterm_diagnostics/dynamic_input_probe_v1`.

Actual dynamic sampling covered 48 distinct clip indices, not all332. Original active termination counts: motion_end6, ee_body_pos23, anchor_pos18 (term counts may overlap). Static counts: ee_body_pos41, motion_end2. These tiny fresh-policy runs are not standing or walking performance acceptance.

| Absolute normalized input | Static p95 / max | Dynamic p95 / max | Dynamic fraction >10 |
|---|---|---|---|
|Proprioception|1.370515 / 3.934577|3.566596 / 24.301748|0.021305|
|Goal|0.354730 / 3.382708|248.250519 / 901.121643|0.561698|
|Critic|0.772376 / 5.363653|223.910019 / 901.121643|0.493317|

The original frozen parent normalizers yield much larger dynamic goal/critic inputs. This is direct range evidence, not proof that normalizers alone cause termination. Next: inspect dimension-wise means/variances and frame construction; establish fit-only normalizers from the original training sources, preserve frozen dimensions, and test a single-variable matched comparison with fixed7 never used for fitting. Keep original-source-normalizer baseline and all failures. Do not increase budgets blindly or change thresholds.


---
## 2026-11-01T12:27Z: training-only goal-normalizer construction audit

Dimension-wise input probe v2 retained prior evidence and independently passed both static/dynamic 1024-transition artifact audits. Parent goal750 dimensions have690 variances below1e-5; critic857 has723. Non-orientation dynamic goal components were amplified while orientation60 dimensions retained nonzero parent variance.

`calibrate_supermimic_training_goal_norm.py` produced an explicitly offline, non-policy calibration from all332 original training references,153960 frames,10 future frames with original terminal clamping. FP64 streaming moments replace690 non-orientation/nonfrozen goal dimensions;60 orientation dimensions remain bit-identical to the source. No simulator steps or fixed7 array reads. Output SHA256 `946bae8cdd4c28cb150b37b545d1d46c36bdd3b95537df9941e29a604fba09c6`; data-disk folder `longterm_diagnostics/training_goal_norm_calibration_v1`. Fitted nearzero variance dimensions54, median variance0.0227950462. Source manifest/isolation/file hashes were checked.

Initial NumPy permutation-invariant joint-moment subset check passed but did not prove name mapping or anchor statistics. The subsequent `audit_supermimic_training_goal_norm.py` independently used original MJCF joint names, TW named default shoulders, all training arrays, NumPy moments and SciPy inverse quaternion transforms for anchor position/linear/angular velocities. It checked all690 replaced dimensions and all60 preserved dimensions. Max mean error5.95017e-8, variance2.97607e-8, below predeclared moment bound1e-6. See `independent_named_audit.json`. This bound audits numerical statistics, not existing physical-control thresholds. No threshold was changed.

Next gate: explicit hash-bound calibration loading for a matched real-PPO experiment, applying the same goal statistics to the actor and critic goal slice only. Preserve all other source statistics/weights, original-normalizer control, fixed training budget and final-update-only evaluation. Run smoke integrity and offline-file rejection checks before full training. This does not establish causal improvement or full SuperMimic acceptance.


---
## 2026-11-01T12:35Z: explicit goal calibration loader and dynamic pilot

Added optional `--goal-normalizer-calibration`; default retains original statistics. The loader verifies kind, parent/manifest/calibration/audit hashes, checked dimensions, frozen masks and preserved orientation. It replaces actor goal mean/variance and critic goal slice73:823 only; other statistics and world-model tensors remain frozen. The independent PPO artifact audit derives the expected calibration explicitly. CPU checks rejected five invalid cases (policy as statistics, wrong parent, wrong manifest, frozen actor goal or critic goal conflicts), with source state unchanged.

First matched smoke retained at `goal_norm_pair_smoke_v1` failed before calibrated training on CPU/CUDA comparison. Fixed by normalizing verification tensors to CPU. Repeated both arms in fresh `goal_norm_pair_smoke_v2`, preserving the failure. Both8-env4-update1024-transition audits passed. Absolute normalized goal p95 baseline266.7903 versus calibrated1.993696, max879.4005 versus9.53727; critic p95239.8965 versus2.063747, fraction>10 0.51449 versus0.00174975. This verifies range correction, not a causal physical benefit. Calibrated tiny-run original term counts motion_end10/ee_body_pos26/anchor_pos17 remain poor.

Predeclared seed42 dynamic pilot: baseline then calibrated, original332 training references and sampler, same original parent with zero action readout,256env x32steps x1000updates =8192000 real transitions per arm, no static sole penalty. Only final update1000 evaluated against original fixed7 windows, trained-only front/side videos and native contacts. Runner preserves failures/checkpoints, records source commit/diff, reserves10GiB, never promotes. Plan: data-disk `longterm_diagnostics/dynamic_goal_norm_pair_v1/plan.json`; script `run_supermimic_dynamic_goal_norm_pair.sh`. Single seed is only a pilot; final>=3 seeds, all fixed7 windows and world-model consistency remain required.

### 2026-11-01: dynamic goal normalization pilot and reference geometry

Both predeclared seed42 arms completed 1000 real PPO updates (8,192,000 transitions), with serialized integrity audits passing. Baseline final SHA256 `39edf12ad653ab833adbc177e0f95eabd3f85517d736766d4f5af6fc31bb08c8`; calibrated final `5f1623915349c7e12e95d14e78999040572e93323fbd1c9fd628f86a25e5edb4`. The unchanged fixed7 acceptance remains **0/7 in both arms**. Baseline/calibrated durations in manifest order: 0.88/0.96, 0.82/1.58, 0.36/0.40, 0.72/1.60, 1.00/1.02, 1.32/1.58, 0.86/0.68 seconds. No checkpoint selection or promotion. All 14 calibrated videos fully decoded and matched trajectory lengths. Front/side first and final frames were visually reviewed for all seven: marked loss of upright balance; sampled review is not full temporal review. Evidence: `dynamic_goal_norm_pair_v1/{calibrated_integrity.json,recorded_action_state_diagnostic.json,calibrated/evaluation/sampled_visual_review.json}` under the remote longterm diagnostics directory.

Read-only reference PD review preserves all 32 source-isolated snippets: 30 original terminations, two short remaining windows. Named action/state order matches, no action clipping, inverse targets checked to 1e-6. This does not establish reference infeasibility: pure joint PD lacks balance feedback.

New `audit_supermimic_reference_support.py` checks all frames of the unchanged eight source-isolated reference motions using original MJCF FK, foot-box corner heights, whole-robot COM, optimistic both-foot XY envelopes and finite-difference COM acceleration. Executed successfully on 3496 frames; FK max below 6.2e-8 m. The optimistic envelope is explicitly **not loaded contact support or a dynamic stability gate**. Three motions contain outside-envelope frames (000112:9, 000152:59, 000203:3); all others zero. This is descriptive geometry, not a failure cause or sample-selection rule. No reference, physics, reward or acceptance thresholds changed. Evidence: `motion_pd_collection_cache_v2/reference_support_geometry_v1.json`. Next: independently check geometric calculations and examine actual/reference loaded support and force/momentum feasibility before another control change. Full SuperMimic world-model and three-seed acceptance remain unfulfilled.

### 2026-11-02: root clearance diagnostic and bounded projection

All-frame reference FK/COM was independently checked using mass-weighted inertial centers and Delaunay containment. For 656/3480 interior frames of the unchanged eight-source collection, both foot-box lowest corners exceed 5 mm while reference COM vertical acceleration requires near body-weight upward external force. All collision mesh/box extrema are above ground in these frames. A compiled one-world TW play environment confirms ground plane at z=0 and zero collision margins/gaps. These observations motivate a reference-contact investigation, not a claim that they alone cause every policy fall.

Original HumanML raw contact marks support at adjacent source samples for all 656 candidates, while conservative known stance exists for only 27; unknown does not imply flight. Side audit does not support a global left/right swap. The raw root-Z projection onto marked support was rejected before generation: 249 frames penetrate the other foot beyond the existing -0.5 mm limit.

The separately declared bounded variant minimizes the scalar marked-foot height target (1 mm) subject to every foot remaining at least 0.2 mm high. Joint positions/velocities, root XY/orientation, body orientations/angular velocities and phase labels remain unchanged. Body positions and linear velocities receive the same vertical translation and its derivative. All eight independent artifacts passed existing known-stance P95 <=3 mm, root shift <=31 mm and full collision minimum >=-0.5 mm. A second all-frame output FK/finite-difference velocity audit passed (FK <6.4e-8 m, velocity translation error <1.38e-6 m/s). Raw-contact errors remain reported separately; the original quality mask is contact AND known. This is geometry/artifact validation only.

`build_supermimic_root_grounding_diagnostic.py` implements the declared transform. Both original and bounded references are being rerun with identical eight sources, four phases, seed42, original PD/caps/termination, maximum100 steps, current-state command-cache refresh, and independent collection audits. Remote state: `reference_root_grounding_pair_v1`. No learned policy, fixed7 modification, strategy promotion or acceptance claim. The raw projection rejection is retained. Git push remains unverified because credentials were unavailable to noninteractive git.

The bounded PD pair completed: both arms terminated 30/32 snippets; bounded improved9/equal18/worse5, total1525→1531 steps (fit +9, validation -3). No significant robust benefit and no larger PPO scheduled from this result. Candidate audit initially failed on absent parent-manifest metadata; failure log retained. Explicit `--root-grounding-parent-plan` now validates unchanged original source selection and isolation, source/derived hashes, immutable arrays and declared translation/velocity correspondence before all original trace audits. Default original audit was rerun and passed; candidate derived audit passed without rerunning collection.

Centroidal wrench diagnostics use full-body inertial centers and inertia, not root acceleration alone. Original reference pressure center exceeds the optimistic both-foot envelope in9 frames of000112 and50 of000152; others remain inside this overly broad envelope, which is not evidence of real loaded support feasibility. `supermimic_reference_wrench.py` independently checked against static/linear-acceleration/principal angular-acceleration analytic mechanics and planar yaw/translation covariance: max errors below3.1e-14 in their units. Reference discretization/contact uncertainties remain.

Same-frame early root/joint comparison (`early_root_joint_divergence.json`): both arms have30 snippets crossing a descriptive10deg root-orientation-error landmark, median14.5steps (~0.29s). At step10 the median root error is5.16/5.29deg, while median joint MAE is0.011229/0.011224rad for original/bounded. The10deg landmark is not a changed termination gate. Evidence suggests root balance departs early despite fairly close initial joint tracking; it does not establish a unique cause or justify modifying robot mass, caps, rewards or test thresholds.
