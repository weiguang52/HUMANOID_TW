#!/bin/bash
# Matched continuation from TW170 control, mixed classes on both GPUs.
set -eo pipefail
GPU=${1:?}; VARIANT=${2:?control or sole}; ITERATIONS=${3:-12000}
MODE=${4:-formal}
[[ "$VARIANT" == control || "$VARIANT" == sole ]]
[[ "$MODE" == formal || "$MODE" == smoke ]]
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/tw176_v1
OUT=$STATE/$VARIANT
[[ "$MODE" == formal ]] || OUT=$STATE/smoke_$VARIANT
mkdir -p "$OUT"
exec 8>"$OUT/train.lock"
flock -n 8
[[ ! -e "$OUT/status" ]] || { echo 'Refusing overwrite'; exit 1; }
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/tw176_v1/train.json
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
FRAMES=$(python -c 'import json,os; print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
TASK=Unitree-Custom-Humanoid-30dof-PathRate-HumanML3D
[[ "$VARIANT" != sole ]] || TASK=Unitree-Custom-Humanoid-30dof-Sole-HumanML3D
RUN_NAME=tw176_mixed_${VARIANT}_${MODE}_s42
EXPERIMENT=unitree_custom_humanoid_30dof_pathrate_humanml3d
LOG_ROOT=logs/rsl_rl/$EXPERIMENT
BASE=$(cat /root/gpufree-data/datasets/practice9/tw170_v1/control/final_checkpoint)
cp /root/gpufree-data/datasets/practice9/tw170_v1/control/final_checkpoint "$OUT/initial_checkpoint"
ENVS=1024
[[ "$MODE" == formal ]] || ENVS=64
if [[ "$MODE" == formal ]]; then
 test -z "$(git status --porcelain)"
 test "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')"
fi
git rev-parse HEAD > "$OUT/source_commit"
python scripts/practice9/audit_tw154_data.py > "$OUT/data_audit.json"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERATIONS" --seed 42 --run_name "$RUN_NAME" --logger tensorboard --resume --load_run "$(basename "$(dirname "$BASE")")" --checkpoint "$(basename "$BASE")" "env.commands.motion.max_motion_frames=$FRAMES" agent.save_interval=500 "agent.experiment_name=$EXPERIMENT" > "$OUT/train.log" 2>&1
RUN=$(find "$LOG_ROOT" -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"
realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo trained > "$OUT/status"
if [[ "$MODE" == formal ]]; then
 bash scripts/practice9/validate_tw176.sh "$GPU" "$VARIANT" > "$OUT/validation.log" 2>&1
 echo completed > "$OUT/status"
fi
