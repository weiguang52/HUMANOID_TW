#!/bin/bash
set -euo pipefail
GPU=${1:?GPU index}; VARIANT=${2:?baseline or resume}; MODE=${3:-formal}
case "$VARIANT" in baseline) KIND=HeadLocked;; resume) KIND=HeadLocked;; *) exit 2;; esac
case "$MODE" in smoke) ENVS=64; ITERATIONS=2;; formal) ENVS=1024; ITERATIONS=395;; *) exit 2;; esac
REPO=/root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/resume_ablation_20261024
OUT="$STATE/${VARIANT}_recovery"
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
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/train.json
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
MEMORY=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( MEMORY < 256 )) || { echo 'GPU occupied'; exit 3; }
if [[ "$MODE" == formal ]]; then
 test "$(cat "$STATE/${VARIANT}_smoke/status")" = trained
fi
INITIAL="$REPO/logs/rsl_rl/unitree_custom_humanoid_30dof_pathrate_humanml3d/2026-10-24_23-57-16_resume_ablation_20261024_${VARIANT}_formal/model_28100.pt"
test -s "$INITIAL"
FRAMES=$(python -c 'import json,os;print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
EXPERIMENT=unitree_custom_humanoid_30dof_pathrate_humanml3d
RUN_NAME=resume_ablation_20261024_${VARIANT}_recovery
git rev-parse HEAD > "$OUT/source_commit"
sha256sum "$INITIAL" "$PRACTICE9_CUSTOM_MOTION_MANIFEST" > "$OUT/input_sha256.txt"
printf '%s\n' "$INITIAL" > "$OUT/initial_checkpoint"
RESUME_ARGS=()
[[ "$VARIANT" != resume ]] || RESUME_ARGS+=(--restore_optimizer_learning_rate)
echo training > "$OUT/status"
python scripts/rsl_rl/train.py "${RESUME_ARGS[@]}" --headless --resume --load_run "$(basename "$(dirname "$INITIAL")")" --checkpoint "$(basename "$INITIAL")" --task "Unitree-Custom-Humanoid-30dof-${KIND}-HumanML3D" --seed 42 --logger tensorboard --num_envs "$ENVS" --max_iterations "$ITERATIONS" --run_name "$RUN_NAME" "env.commands.motion.max_motion_frames=$FRAMES" "agent.experiment_name=$EXPERIMENT" agent.save_interval=100 > "$OUT/train.log" 2>&1
RUN=$(find "logs/rsl_rl/$EXPERIMENT" -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"; realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo trained > "$OUT/status"
