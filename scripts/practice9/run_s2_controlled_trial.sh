#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
GPU="${1:?GPU required}"
VARIANT="${2:?baseline or smooth required}"
case "$VARIANT" in baseline) RATE=-0.05;; smooth) RATE=-0.2;; *) exit 2;; esac
OUT="$DATA/datasets/practice9/s2_trials/$VARIANT"
mkdir -p "$OUT"
trap 'code=$?; if ((code)); then echo failed:$code > "$OUT/status"; fi' EXIT
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_simple_v1/curriculum.json"
unset P9_CUSTOM_ALLOW_UNSAFE_MOTIONS
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
python - "$PRACTICE9_CUSTOM_MOTION_MANIFEST" <<'PYCHECK'
import sys
from pathlib import Path
sys.path.insert(0,'scripts/practice9')
from run_native_motion_pipeline import require_manifest
require_manifest(Path(sys.argv[1]),training=True,expected_backend='s2_curriculum_v1')
PYCHECK
CHECKPOINT=$(cat "$DATA/datasets/practice9/tw59_s1/checkpoint")
LOAD_RUN=$(basename "$(dirname "$CHECKPOINT")")
git rev-parse HEAD > "$OUT/source_commit"
cp "$PRACTICE9_CUSTOM_MOTION_MANIFEST" "$OUT/training_manifest.json"
printf '%s\n' "$RATE" > "$OUT/action_rate_weight"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1024 --max_iterations 1000 --seed 42 --run_name "s2_trial_$VARIANT" --logger tensorboard --resume --load_run "$LOAD_RUN" --checkpoint model_9999.pt "env.rewards.action_rate_l2.weight=$RATE" > "$OUT/train.log" 2>&1
echo trained > "$OUT/status"
