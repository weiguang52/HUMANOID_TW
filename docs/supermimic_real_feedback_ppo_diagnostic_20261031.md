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
