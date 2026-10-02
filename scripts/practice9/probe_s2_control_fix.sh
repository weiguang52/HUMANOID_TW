#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
GPU="$1"; NAME="$2"; TAU="$3"; LIMIT="$4"
STATE="$DATA/datasets/practice9/s2_control_fix/$NAME"
mkdir -p "$STATE"
export CUDA_VISIBLE_DEVICES="$GPU" P9_TARGET_SMOOTHING_TAU="$TAU" P9_TARGET_LIMIT_VELOCITY="$LIMIT"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
trap 'code=$?; if ((code)); then echo failed:$code > "$STATE/status"; fi' EXIT
echo running > "$STATE/status"
CHECKPOINT=$(cat "$DATA/datasets/practice9/s2_trials/smooth/final_checkpoint")
for ID in ${S2_PROBE_IDS:-000801}; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_trials/smooth/$ID.manifest.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed 42 --disable_observation_noise --evaluation_output "$STATE/$ID.evaluation.json" --telemetry_output "$STATE/$ID.telemetry.npz" > "$STATE/$ID.log" 2>&1
done
echo completed > "$STATE/status"
