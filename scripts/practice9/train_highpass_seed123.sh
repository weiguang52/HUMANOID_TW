#!/bin/bash
set -euo pipefail
GPU=${1:?GPU index}; VARIANT=${2:?moving}; MODE=${3:-formal}
[[ "$VARIANT" == moving ]] || { echo "Use the recorded matched baseline from filtered_moving_slip_20261025"; exit 2; }
case "$VARIANT" in baseline) KIND=SlipBaseline;; moving) KIND=MovingSlip;; *) exit 2;; esac
case "$MODE" in probe) ENVS=64; ITERATIONS=20;; smoke) ENVS=64; ITERATIONS=2;; formal) ENVS=1024; ITERATIONS=500;; *) exit 2;; esac
REPO=/root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/highpass_seed123_20261025
OUT="$STATE/${VARIANT}_${MODE}"
mkdir -p "$OUT"
exec 8>"$OUT/run.lock"; flock -n 8
[[ ! -e "$OUT/status" ]] || { echo 'Refusing to overwrite an existing run'; exit 1; }
exec > >(tee -a "$OUT/supervisor.log") 2>&1
trap 'echo failed > "$OUT/status"' ERR
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
set -u
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/highpass_seed123_20261025/train.json
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
MEMORY=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( MEMORY < 256 )) || { echo 'GPU occupied'; exit 3; }
if [[ "$MODE" == formal ]]; then
 test "$(cat "$STATE/${VARIANT}_smoke/status")" = trained
fi
INITIAL=$(cat /root/gpufree-data/datasets/practice9/head_locked_20261024/final_checkpoint)
test -s "$INITIAL"
FRAMES=$(python -c 'import json,os;print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
EXPERIMENT=unitree_custom_humanoid_30dof_pathrate_humanml3d
RUN_NAME=highpass_seed123_20261025_${VARIANT}_${MODE}
git rev-parse HEAD > "$OUT/source_commit"
sha256sum "$INITIAL" "$PRACTICE9_CUSTOM_MOTION_MANIFEST" > "$OUT/input_sha256.txt"
printf '%s\n' "$INITIAL" > "$OUT/initial_checkpoint"
RESUME_ARGS=(--physics_mean_slip --train_ankle_filter_alpha 0.5 --restore_optimizer_learning_rate --policy_anchor_coefficient 1.0 --policy_anchor_checkpoint "$INITIAL")

sha256sum source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/leg_error_highpass.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/leg_velocity_error.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/point_slip.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/custom_30dof/humanml3d/point_slip_env_cfg.py scripts/rsl_rl/train.py "$0" > "$OUT/code_sha256.txt"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py "${RESUME_ARGS[@]}" --headless --resume --load_run "$(basename "$(dirname "$INITIAL")")" --checkpoint "$(basename "$INITIAL")" --task "Unitree-Custom-Humanoid-30dof-${KIND}-HumanML3D" --seed 123 --logger tensorboard --num_envs "$ENVS" --max_iterations "$ITERATIONS" --run_name "$RUN_NAME" "env.commands.motion.max_motion_frames=$FRAMES" "agent.experiment_name=$EXPERIMENT" agent.save_interval=100 env.rewards.moving_point_slip.weight=-1.0 env.rewards.action_rate_l2.weight=-0.3 env.rewards.moving_leg_velocity.weight=-0.6 env.rewards.moving_leg_highpass.weight=-0.2 > "$OUT/train.log" 2>&1
RUN=$(find "logs/rsl_rl/$EXPERIMENT" -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"; realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo trained > "$OUT/status"
