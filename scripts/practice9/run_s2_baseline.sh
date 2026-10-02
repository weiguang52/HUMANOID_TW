#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
OUT="$DATA/datasets/practice9/s2_diagnostics/baseline"
mkdir -p "$OUT"
trap 'code=$?; if (( code != 0 )); then echo failed:$code > "$OUT/status"; fi' EXIT
export CUDA_VISIBLE_DEVICES=1 ISAACLAB_PATH="$DATA/projects/IsaacLab"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
CHECKPOINT=$(cat "$DATA/datasets/practice9/tw59_s1/checkpoint")
echo running > "$OUT/status"
for SEED in 42 43 44 45 46; do
 for ID in 000801 006680 010407 010782; do
  export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/tw59_s1/$ID.manifest.json"
  LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
  python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed "$SEED" --telemetry_output "$OUT/${ID}_seed${SEED}.npz" --evaluation_output "$OUT/${ID}_seed${SEED}.evaluation.json" > "$OUT/${ID}_seed${SEED}.log" 2>&1
 done
done
echo completed > "$OUT/status"
