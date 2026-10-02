#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
GPU="$1"; VARIANT="$2"; ACC="$3"
OUT="$DATA/datasets/practice9/s2_control_fix/$VARIANT"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU" P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_simple_v1/curriculum.json"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
trap 'code=$?; if ((code)); then echo failed:$code > "$OUT/status"; fi' EXIT
CHECKPOINT=$(cat "$DATA/datasets/practice9/s2_trials/smooth/final_checkpoint")
git rev-parse HEAD > "$OUT/source_commit"
git diff > "$OUT/source.patch"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1024 --max_iterations 600 --seed 42 --run_name "s2_fix_$VARIANT" --logger tensorboard --resume --load_run "$(basename "$(dirname "$CHECKPOINT")")" --checkpoint "$(basename "$CHECKPOINT")" env.rewards.action_rate_l2.weight=-0.2 "env.rewards.joint_acc.weight=$ACC" > "$OUT/train.log" 2>&1
RUN=$(find "$REPO/logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d" -maxdepth 1 -type d -name "*_s2_fix_$VARIANT" | sort | tail -1)
find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1 > "$OUT/final_checkpoint"
echo trained > "$OUT/status"
# Replay restores the exact target/actuator contract exported by this run.
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT
CHECKPOINT=$(cat "$OUT/final_checkpoint")
echo evaluating > "$OUT/validation.status"
for ID in 000801 006680 010407 010782 s2_stand s2_weight_shift s2_wave_000113; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_trials/smooth/$ID.manifest.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed 42 --disable_observation_noise --evaluation_output "$OUT/$ID.evaluation.json" --telemetry_output "$OUT/$ID.telemetry.npz" > "$OUT/$ID.log" 2>&1
done
echo completed > "$OUT/validation.status"
