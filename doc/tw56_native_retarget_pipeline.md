# TW-56: Native retargeting and the RL pipeline

The active backend is https://github.com/weiguang52/tw_retargeting at
`087697e9cdcc3ff6516479a2562fdfa5cb2d96674` (latest default HEAD when integrated).
The checkout lives at `/root/gpufree-data/projects/tw_retargeting`; the native
project is `IKRetargeting_V3`. No IK_V2/Pink solver is imported or used by the
active retarget entry point. Historical datasets and code outside this repository
are retained on disk, but training defaults no longer select their manifests.

## Interface

1. Read finite HumanML3D 20 Hz NPY `[T>=9,J>=22,3]`. Extended 43-joint input is
   supported; native IK uses the first 22 positions.
2. Run the upstream selected C++ head-axis configuration, one IK iteration,
   stream mode, with a reset per independent motion. Bare native CLI defaults
   are not substituted for the selected configuration.
3. Validate native 28-joint output and QP KKT, resample 90 Hz to 50 Hz on exact
   physical timestamps, then map the reference 28-DoF coordinates to the TW-44
   aligned 30-DoF robot. Added foot rolls remain zero.
4. Restore root translation/yaw, apply unchanged position/velocity/root-motion
   quality checks and bounded automatic time scaling.
5. Compute and validate Isaac FK and body velocities, then run PPO with the
   accepted native motion manifest.

NPZ and manifest metadata record upstream commit, binary/config/solver-URDF
hashes, native joint order, selected flags and native timing. Already aligned
motion files are not coordinate-converted again. Limits and quality checks were
not relaxed to increase acceptance. The 28-to-30 mapping remains bootstrap:
reference-model dimensions and waist ordering differ from the training model.

## Environment and build

All commands run on remote Linux, from `/root/gpufree-data/projects/HUMANOID_TW`:

```bash
conda activate /root/gpufree-data/conda_envs/env_isaaclab
bash scripts/practice9/build_native_retarget.sh
```

The server's existing Pinocchio 2.7.0 C++ libraries are reused. Eigen 3.4.1 and
urdfdom-headers 3.0.0 are installed into the native project's ignored
`build/deps` directory, leaving the Isaac Python environment unchanged.
`PRACTICE9_RETARGET_ROOT` overrides the native project path for the build script;
`--retarget-root` does the same for the pipeline/retarget CLI.

## One command and resumable stages

The input may be an NPY, directory, or text list of NPY paths. Text-list paths
can be absolute or relative to the list file; duplicate filename stems are
rejected. No full-dataset run is started implicitly by the unified CLI.

```bash
python scripts/practice9/run_native_motion_pipeline.py --stage all \
  --input /root/gpufree-data/datasets/practice9/tw56_native_v1/approved_inputs.txt \
  --output-root /root/gpufree-data/datasets/practice9/tw56_native_smoke \
  --num-envs 2 --max-iterations 1 --run-name tw56_full_pipeline_smoke
```

Available stages are `build`, `retarget`, `fk`, `train`, and `all`. Retarget
and all require explicit input; FK and train reuse their stage manifests.
The default output root is `/root/gpufree-data/datasets/practice9/tw56_native_v1`.
It contains `retargeted/`, `retarget_manifest.json`, `npz/`, and `manifest.json`.
The initial 165-candidate source list is `inputs.txt`; `approved_inputs.txt`
contains the four source clips accepted during the integration check.

Normal training now selects the native manifest by default and rejects legacy,
empty, non-50-Hz, wrong-coordinate or quality-failed manifests before launch:

```bash
python scripts/practice9/run_native_motion_pipeline.py --stage train \
  --num-envs 1024 --max-iterations 10000 --run-name tw56_native_training
```

GPU defaults to physical device 1; use `--gpu` to select another. The training
launcher retains its GPU occupancy guard. Old policies should not be resumed
unchanged across the TW-44 action-coordinate change. This integration validates
a short training run, rather than policy convergence.

## Verification

- Upstream native build completed; all four C++ tests passed.
- Real 22-position and extended 43-position inputs produced identical IK output.
- 78 Python tests plus eight subtests passed, including native resampling,
  timestamps, input checks and legacy-manifest rejection.
- Of 165 locomotion candidates, four passed unchanged retarget quality checks;
  the other 161 were rejected, primarily for sustained ankle-yaw limit proximity.
- All four accepted clips passed Isaac FK conversion: 3,746 frames at 50 Hz,
  IDs 000801, 006680, 010407 and 010782.
- Native-data PPO smoke completed with two environments and one iteration.
- Full build -> retarget -> FK -> PPO verification completed successfully;
  logs are `/root/gpufree-data/tmp/tw56-full-pipeline.log`.

No files from the mounted server were copied to the local computer.
