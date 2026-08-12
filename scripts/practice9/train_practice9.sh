#!/usr/bin/env bash
set -eo pipefail
P9_NUM_ENVS="${P9_NUM_ENVS:-2048}"
P9_MAX_ITERATIONS="${P9_MAX_ITERATIONS:-20000}"
P9_SEED="${P9_SEED:-42}"
P9_RUN_NAME="${P9_RUN_NAME:-practice9_dance1_gpu1}"
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
ENV_ISAACLAB="$DATA/conda_envs/env_isaaclab"
export ISAACLAB_PATH="$DATA/projects/IsaacLab"
export PRACTICE9_MOTION_FILE="$DATA/datasets/practice9/dance1_subject2.npz"
export CUDA_VISIBLE_DEVICES=1
export PIP_CACHE_DIR="$DATA/.cache/pip"
export XDG_CACHE_HOME="$DATA/.cache/xdg"
export CUDA_CACHE_PATH="$DATA/.cache/nvidia/practice9"
export OMNI_USER_DIR="$DATA/.cache/omniverse/practice9"
export WANDB_DIR="$DATA/wandb/practice9"
export WANDB_CACHE_DIR="$DATA/.cache/wandb"
export WANDB_ARTIFACT_DIR="$DATA/wandb/practice9/artifacts"
export TMPDIR="$DATA/tmp/practice9"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$OMNI_USER_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"
test -s "$PRACTICE9_MOTION_FILE" || { echo "Missing motion: $PRACTICE9_MOTION_FILE" >&2; exit 1; }
gpu1_memory=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "$gpu1_memory" =~ ^[0-9]+$ ]] || (( gpu1_memory > 256 )); then
  echo "Refusing to start: physical GPU 1 is already occupied." >&2
  exit 1
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_ISAACLAB"
set -u
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
cd "$REPO"
num_envs="$P9_NUM_ENVS"
max_iterations="$P9_MAX_ITERATIONS"
seed="$P9_SEED"
run_name="$P9_RUN_NAME"
exec python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Mimic-Dance-102 --num_envs "$num_envs" --max_iterations "$max_iterations" --seed "$seed" --run_name "$run_name" "$@"
