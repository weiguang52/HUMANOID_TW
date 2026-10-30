#!/bin/bash
set -eo pipefail
STATE=$1; VIEW=$2; GPU=$3; ROOT=/root/gpufree-data/datasets/practice9
D=$STATE/videos/retention/$VIEW
trap 'echo failed > "$D/status"' ERR
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=.5 P9_CONTACT_POINT_TELEMETRY=1
unset P9_LOCOMOTION_DAMPING P9_PD_DAMPING_MULT P9_PD_STIFFNESS_MULT P9_ANKLE_YAW_DAMPING_MULT
unset P9_TARGET_VELOCITY_SCALE P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=$ROOT/tw198_v1/manifest.json
echo running > "$D/status"
python scripts/rsl_rl/play.py --headless --enable_cameras --robot_usd "$ROOT/head_locked_20261024/robot_colored.usd" --task Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D --num_envs 1 --checkpoint "$(cat "$STATE/checkpoint")" --seed 42 --standing_diagnostic --evaluation_batch "$D/jobs.json" > "$D/run.log" 2>&1
test "$(cat "$D/jobs.json.done")" = completed
echo completed > "$D/status"
