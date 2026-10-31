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
