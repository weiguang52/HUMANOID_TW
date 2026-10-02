#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
GPU="$1"; VARIANT="$2"
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
trap 'code=$?; if ((code)); then echo failed:$code > "$OUT/status"; echo failed:$code > "$OUT/validation.status"; fi' EXIT
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT
CHECKPOINT=$(cat "$OUT/final_checkpoint")
echo robustness_evaluation > "$OUT/status"
for MODE in seed123 seed2026 noise123; do
 SEED="${MODE#seed}"; NOISE=(--disable_observation_noise)
 if [[ "$MODE" == noise123 ]]; then SEED=123; NOISE=(); fi
 mkdir -p "$OUT/$MODE"
 for ID in 000801 006680 010407 010782 s2_stand s2_weight_shift s2_wave_000113; do
  export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_trials/smooth/$ID.manifest.json"
  LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
  if python - "$OUT/$MODE/$ID" "$LENGTH" "$CHECKPOINT" <<'CHECK'
import sys,json,numpy as np
from pathlib import Path
p=Path(sys.argv[1]);e=json.load(open(str(p)+'.evaluation.json'))
assert e['steps']==int(sys.argv[2]) and e['checkpoint']==sys.argv[3]
with np.load(str(p)+'.telemetry.npz') as d:
 assert len(d['action'])==int(sys.argv[2])
CHECK
  then echo "Verified existing $MODE/$ID"; continue; fi
  python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed "$SEED" "${NOISE[@]}" --evaluation_output "$OUT/$MODE/$ID.evaluation.json" --telemetry_output "$OUT/$MODE/$ID.telemetry.npz" > "$OUT/$MODE/$ID.log" 2>&1
 done
done
echo completed > "$OUT/status"
