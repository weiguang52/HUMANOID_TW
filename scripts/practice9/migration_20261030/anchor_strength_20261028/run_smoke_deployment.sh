#!/bin/bash
set -eo pipefail
D=/root/gpufree-data/datasets/practice9/anchor_strength_20261028
ROOT=/root/gpufree-data/datasets/practice9
exec > "$D/smoke_deployment/supervisor.log" 2>&1
trap 'echo failed > "$D/smoke_deployment/status"' ERR
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=.5 P9_CONTACT_POINT_TELEMETRY=1
unset P9_LOCOMOTION_DAMPING P9_PD_DAMPING_MULT P9_PD_STIFFNESS_MULT P9_ANKLE_YAW_DAMPING_MULT
unset P9_TARGET_VELOCITY_SCALE P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$ROOT/training_speed_blend_20261026/evaluation/manifest.json"
CKPT=$(cat "$D/actual_smoke/final_checkpoint")
python -c 'import json,sys,hashlib;assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()==json.load(open(sys.argv[2]))["checkpoint_sha256"]' "$CKPT" "$D/smoke_audit.json"
echo running > "$D/smoke_deployment/status"
python scripts/rsl_rl/play.py --headless --robot_usd "$ROOT/head_locked_20261024/robot_colored.usd" --task Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D --num_envs 1 --checkpoint "$CKPT" --seed 42 --disable_observation_noise --evaluation_upper_body_termination --evaluation_batch "$D/smoke_deployment/jobs.json" > "$D/smoke_deployment/run.log" 2>&1
test "$(cat "$D/smoke_deployment/jobs.json.done")" = completed
echo completed > "$D/smoke_deployment/status"
