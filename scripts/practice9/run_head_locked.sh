#!/bin/bash
set -euo pipefail
REPO=/root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/head_locked_20261024
mkdir -p "$STATE"
exec 7>"$STATE/run.lock"; flock -n 7
[[ ! -e "$STATE/status" ]] || { echo 'Refusing to overwrite existing run'; exit 1; }
exec > >(tee -a "$STATE/supervisor.log") 2>&1
trap 'echo failed > "$STATE/status"' ERR
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
set -u
cd "$REPO"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/cpp_refresh_20261024/train.json
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
FRAMES=$(python -c 'import json,os;print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
INITIAL=$(cat /root/gpufree-data/datasets/practice9/cpp_refresh_20261024/final_checkpoint)
test -s "$INITIAL"
EXPERIMENT=unitree_custom_humanoid_30dof_pathrate_humanml3d
COMMON=(--headless --resume --load_run "$(basename "$(dirname "$INITIAL")")" --checkpoint "$(basename "$INITIAL")" --task Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D --seed 42 --logger tensorboard "env.commands.motion.max_motion_frames=$FRAMES" "agent.experiment_name=$EXPERIMENT")
echo smoke > "$STATE/status"
python scripts/rsl_rl/train.py "${COMMON[@]}" --num_envs 64 --max_iterations 2 --run_name head_locked_20261024_smoke > "$STATE/smoke.log" 2>&1
echo training > "$STATE/status"
git rev-parse HEAD > "$STATE/source_commit"
python scripts/rsl_rl/train.py "${COMMON[@]}" --num_envs 1024 --max_iterations 2000 --run_name head_locked_20261024_formal agent.save_interval=500 > "$STATE/train.log" 2>&1
RUN=$(find "logs/rsl_rl/$EXPERIMENT" -maxdepth 1 -type d -name '*_head_locked_20261024_formal' | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT";realpath "$CHECKPOINT" > "$STATE/final_checkpoint"
echo trained > "$STATE/status"
