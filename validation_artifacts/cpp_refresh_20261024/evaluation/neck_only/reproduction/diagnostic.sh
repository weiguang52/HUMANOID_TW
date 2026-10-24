#!/bin/bash
set -eo pipefail
GPU=${1:?}; VARIANT=${2:?}
case "$VARIANT" in
 baseline) KIND=PathRate;; neck_only) KIND=NeckOnly;; *) exit 2;;
esac
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/evaluation
OUT=$STATE/validation/$VARIANT
trap 'echo failed > "$OUT/diagnostic.status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=0.5
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/validation/$VARIANT/manifest.json"
CHECKPOINT=$(cat "$STATE/$VARIANT/final_checkpoint")
TASK=Unitree-Custom-Humanoid-30dof-${KIND}-HumanML3D
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/tw198_v1/manifest.json
echo synthetic_diagnostics > "$OUT/diagnostic.status"
python scripts/practice9/tw108_remaining_jobs.py "$STATE/diagnostic/$VARIANT/jobs.json"
if [[ ! -s "$STATE/diagnostic/$VARIANT/jobs.json.remaining.done" ]]; then
 timeout --signal=TERM --kill-after=15s 1800s python scripts/rsl_rl/play.py --headless --task "$TASK" --num_envs 1 --checkpoint "$CHECKPOINT" --seed 42 --standing_diagnostic --enable_cameras --evaluation_batch "$STATE/diagnostic/$VARIANT/jobs.json.remaining" > "$OUT/diagnostic.log" 2>&1
fi
echo completed > "$OUT/diagnostic.status"
