#!/bin/bash
# Stage one only. Never automatically promote to walking before validation.
set -eo pipefail
GPU=${1:?GPU}; VARIANT=${2:?expression or control}; ITERATIONS=${3:-6000}
[[ "$VARIANT" == expression || "$VARIANT" == control ]]
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/tw100_expression_v1
OUT=$STATE/$VARIANT
mkdir -p "$OUT"
[[ ! -e "$OUT/status" ]] || { echo 'Refusing to overwrite existing training'; exit 1; }
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/standing_train.json"
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
FRAMES=$(python -c 'import json,os; print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
TASK=Unitree-Custom-Humanoid-30dof-Expression-HumanML3D
[[ "$VARIANT" != control ]] || TASK=Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D
RUN_NAME=tw100_standing_${VARIANT}_s42
LOG_ROOT=logs/rsl_rl/$(echo "$TASK" | tr '[:upper:]-' '[:lower:]_')
git rev-parse HEAD > "$OUT/source_commit"
sha256sum "$PRACTICE9_CUSTOM_MOTION_MANIFEST" "$STATE/standing_train.contacts.npz" > "$OUT/data_sha256"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task "$TASK" --num_envs 1024 --max_iterations "$ITERATIONS" --seed 42 --run_name "$RUN_NAME" --logger tensorboard env.rewards.action_rate_l2.weight=-0.2 env.rewards.joint_acc.weight=-2e-6 env.rewards.motion_joint_vel_cost.weight=-0.2 env.commands.motion.adaptive_uniform_ratio=1.0 env.commands.motion.adaptive_max_probability=null "env.commands.motion.max_motion_frames=$FRAMES" agent.save_interval=500 > "$OUT/train.log" 2>&1
RUN=$(find "$LOG_ROOT" -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"
realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo trained_validation_pending > "$OUT/status"
