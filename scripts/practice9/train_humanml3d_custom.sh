#!/usr/bin/env bash
set -euo pipefail

DATA="${GPUFREE_DATA_ROOT:-/root/gpufree-data}"
REPO="$DATA/projects/HUMANOID_TW"
ENV_ISAACLAB="$DATA/conda_envs/env_isaaclab"
GPU="${P9_CUSTOM_GPU:-1}"
NUM_ENVS="${P9_CUSTOM_NUM_ENVS:-1024}"
MAX_ITERATIONS="${P9_CUSTOM_MAX_ITERATIONS:-10000}"
SEED="${P9_CUSTOM_SEED:-42}"
RUN_NAME="${P9_CUSTOM_RUN_NAME:-practice9_humanml3d_custom30}"
MANIFEST="${PRACTICE9_CUSTOM_MOTION_MANIFEST:-$DATA/datasets/practice9/humanml3d_custom30_locomotion400_v1/manifest.json}"

export GPUFREE_DATA_ROOT="$DATA"
export ISAACLAB_PATH="$DATA/projects/IsaacLab"
export PRACTICE9_CUSTOM_URDF="${PRACTICE9_CUSTOM_URDF:-$DATA/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf}"
export PRACTICE9_CUSTOM_USD_DIR="${PRACTICE9_CUSTOM_USD_DIR:-$DATA/datasets/practice9/custom_robot/usd}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$MANIFEST"
export CUDA_VISIBLE_DEVICES="$GPU"
export PIP_CACHE_DIR="$DATA/.cache/pip"
export XDG_CACHE_HOME="$DATA/.cache/xdg"
export CUDA_CACHE_PATH="$DATA/.cache/nvidia/practice9_custom"
export OMNI_USER_DIR="$DATA/.cache/omniverse/practice9_custom"
export WANDB_DIR="$DATA/wandb/practice9_custom"
export WANDB_CACHE_DIR="$DATA/.cache/wandb"
export WANDB_ARTIFACT_DIR="$DATA/wandb/practice9_custom/artifacts"
export TMPDIR="$DATA/tmp/practice9_custom"

mkdir -p   "$PRACTICE9_CUSTOM_USD_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"   "$CUDA_CACHE_PATH" "$OMNI_USER_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"   "$WANDB_ARTIFACT_DIR" "$TMPDIR"

set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_ISAACLAB"
set -u
python "$REPO/scripts/practice9/prepare_custom_robot_urdf.py" --output "$PRACTICE9_CUSTOM_URDF" >/dev/null
test -s "$PRACTICE9_CUSTOM_URDF" || { echo "Missing training URDF: $PRACTICE9_CUSTOM_URDF" >&2; exit 2; }
test -s "$MANIFEST" || { echo "Missing motion manifest: $MANIFEST" >&2; exit 2; }

gpu_memory="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$gpu_memory" =~ ^[0-9]+$ ]] || (( gpu_memory > 256 )); then
  echo "Refusing to start: physical GPU $GPU is already occupied (${gpu_memory} MiB)." >&2
  exit 3
fi

export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
cd "$REPO"
exec python scripts/rsl_rl/train.py   --headless   --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D   --num_envs "$NUM_ENVS"   --max_iterations "$MAX_ITERATIONS"   --seed "$SEED"   --run_name "$RUN_NAME"   "$@"
