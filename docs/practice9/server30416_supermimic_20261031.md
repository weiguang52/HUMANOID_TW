# Server30416 migration and initial TW SuperMimic adapter

Status: original Isaac training configuration verified; SuperMimic full package tests, training and effects are still pending. This document does not promote any policy.

## Storage and original training
All project files, environments, models, motion data, temporary files and runtime caches belong under /root/gpufree-data. Read dependencies from the existing /opt/conda bootstrap only. Source scripts/practice9/activate_data_environment.sh before original Isaac runs. The isolated migration check launcher supports smoke/reload and refuses overwriting existing outputs.

The migration copied the datasets, Conda environments, Isaac Sim/Lab, model logs, old experiment states, meshes and reference videos directly between servers. Initial missing links were repaired with explicit receipts; no files were copied through the local computer. Content evidence is in /root/gpufree-data/migration/combined_content_audit.json and its original source manifests and failed-entry rechecks. Initial failures remain preserved. Original scope:509267 entries, legacy14376, validation artifacts6434, plus43 mesh and198 importer-extension files checked separately.

Original Torch2.7.0+cu128/CUDA12.8 GPU computation passed. Original Isaac64-environment smoke completed2 PPO iterations,48 control windows,Adam560040; reload completed1 iteration,24 windows,Adam560060. Checkpoint/state, finite weights,195-to203 actor input expansion and real weight updates are checked by audit_migration_training.py. Results: datasets/practice9/migration_smoke_20261031/{smoke,reload}_migration_audit.json. This is environment/recovery proof, not policy quality or convergence.

## Method and source
Paper: https://arxiv.org/abs/2610.28480 . Author project: https://supermimic.github.io/ . The vendored Apache-2.0 implementation is third-party YHFone/supermimic-mjlab, commit9aab0a588c6941bf9862a2d444b711fd73e2b832, not official author source. Preserve LICENSE and TW_UPSTREAM.md. Original downloaded36 files were checked against Git blob hashes. Local changes are tracked by this repository.

SuperMimic alternates simulator data collection, multi-step world-model fitting and actor BPTT through a frozen-within-update world model with a critic bootstrap. The world model updates each iteration; it is not permanently frozen. The initial adapter preserves this algorithm loop.

## Current robot work and evidence
- Explicit TW30-joint contract,14 tracking bodies,chest anchor,shoulder roll readout offsets and original PD/effort/armature/action scales. Original action ordering is resolved by names.
- Context torso/foot geometry names are configurable; empty foot matches fail explicitly instead of generating NaNs.
- TW motion input requires named joint/body metadata,current tw123_yaw_waist_v3 coordinates,finite arrays,valid quaternions and consistent FPS. Positions and velocities both map by joint name. train_tw supports the existing train.json manifest without dropping missing clips.
- STL exceeds MuJoCo's200k-face decoder limit. stl_to_obj_exact preserves every oriented float32 triangle, checks the reread OBJ against the original triangle stream and does not decimate the asset. Original URDF stays unchanged.
- MJCF export restores full numeric precision instead of accepting default XML rounding. Actual MuJoCo3.11 compile passes30 joints,nq37/nv36,mass0.840742808746342kg and joint/mass checks.
- Independent URDF algebra versus reloaded MJCF,50 random poses/all35 links: max position1.665e-16m,rotation-matrix9.992e-16,inertia6.242e-12,COM0. All332 training clips/153960 frames compared on14 tracking bodies: max position1.076e-7m,orientation9.215e-7rad.
- CPU fixed-target PD/contact diagnostic20s is finite,torque-limited,3997/4000 contact steps,about2.1mm XY root displacement. This is not trained-policy evaluation. Velocity caps have not yet been enforced or validated under motion/exploration.

Evidence lives under /root/gpufree-data/projects/supermimic_research and datasets/practice9/supermimic_tw_20261031/robot. tw30_precision.xml has the strict asset audit; tw30.xml is reserved for the complete-environment checks.

## Independent runtime and next actions
Source scripts/practice9/activate_supermimic_environment.sh for the separate env_supermimic environment and vendored package path. It clears old runtime library paths. Torch2.7/CUDA12.8 was copied independently from the verified data-volume runtime; no mutable hardlinks or symlinks point into the original environment. Direct GPU computation with PYTHONPATH/PYTHONHOME/LD_LIBRARY_PATH removed passed. Other dependencies and pip check are pending. Do not call the full package installed until these checks complete.

The unconstrained install initially selected Torch2.15/CUDA13.2; the intended runtime is fixed to Torch2.7.0+cu128,mjlab1.6.0,mujoco3.11.0,warp1.18.0. Installer and initial errors are retained in the research state. The original Isaac environment is not modified.

Next: finish independent dependency installation/pip check; run unchanged upstream tests and the local adapter tests; compare differentiable FK with MuJoCo,assemble the actual TW Entity and prove action order/PD/effort/armature/contact/velocity behavior; run the complete collect-fit-imagine loop and checkpoint reload on actual TW motions. Then train a meaningful pilot,compare simulator performance and world-model errors,check standing/arms/turning and walking fidelity,and deliver colored front/side plus original HumanML3D skeleton videos. Preserve source timing and label any retarget timing differences. Cross-simulator friction and velocity limits are not yet equivalent to original PhysX. Reward scale and dynamic consistency must be measured,not assumed from G1 defaults. No overall acceptance or automatic promotion before actual results.

---
## Runtime and full-method verification (2026-10-31 CST)

Independent runtime installation completed with mjlab 1.6.0, MuJoCo 3.11.0,
Warp 1.18.0 and Torch 2.7.0+cu128; `pip check` reports no broken requirements.
Upstream fast tests: 20; adapted fork: initially 27, then 29 including speed-cap
projection checks. Actual TW Entity construction also exposed/fixed a required
collision priority. TW contacts needed nconmax=128/njmax=512 (59 initial
contacts exceeded G1's nconmax=35).

The first actual Warp zero-action check reached 7.9053 rad/s despite original
PhysX group caps of 2 or 3 rad/s. TW now explicitly projects the integrated joint
velocity before position integration, using a per-simulation Warp acceleration
scratch array; floating-root acceleration is unchanged. This hook checks the
installed integrator source fingerprint, requires implicitfast, and recaptures
CUDA graphs. Other unregistered simulations retain upstream integration.
This is an explicit engine adaptation, not a claim of identical PhysX contact
or constraint solving. In 8 environments/100 control steps, all 30 per-joint
caps passed and peak speed was 3.0 rad/s. Zero actions still caused 42 termination
events, so this test does not establish policy quality.

A real collect/fit/imagine smoke used all 332 clips/153960 frames, 32 environments,
H=32 and world-model window=8, full network widths, reduced batches and four
iterations. All losses were finite, bad actor gradients were zero, and a
checkpoint-to-checkpoint audit confirmed actual encoder, decoder, critic and
world-model parameter updates. Adam steps progressed WM 4->8, actor 2->4 and
critic 8->16 between iterations 2 and 4. This is execution evidence, not learning
convergence. Checkpoint SHA256:
`32bed35c595f30bab1164abc1db4d9d0198ee13ae7330278557624125e351c3c`.

The fork now saves/restores optimizer states, RNG state and fractional collection
step accumulator, rejects missing optimizer states and requires strict parameter
loading. Replay, simulator state and return-sampler statistics are rebuilt on
resume; continuation is not bit-identical replay. TW production entrypoint also
records/checks coordinate, MJCF, ordered-motion-input and speed-limit contracts.

Reload passed: iterations 4->6, WM Adam 8->12, actor 4->6, critic 16->24, all parameter groups changed and remained finite. Reload checkpoint SHA256: `889424fb666b6491a2f451faf011ae16670046f9e521538b6def4886339482e9`.

Pending: bounded 1000-iteration method-transfer pilot (256
envs, WM batch512/10 updates, policy batch512/chunk128, unchanged H32/window8),
fixed test-set evaluation, source-reference plus colored front/side videos and
visual checks. No SuperMimic quality acceptance or policy promotion yet.
