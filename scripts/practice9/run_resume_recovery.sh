#!/bin/bash
set -euo pipefail
GPU=${1:?}; VARIANT=${2:?}
[[ "$VARIANT" == baseline || "$VARIANT" == resume ]] || exit 2
REPO=/root/gpufree-data/projects/HUMANOID_TW
ROOT=/root/gpufree-data/datasets/practice9/resume_ablation_20261024
E="$ROOT/$VARIANT/evaluation_colored"
cd "$REPO"
exec 7>"$ROOT/$VARIANT/experiment.lock"; flock -n 7
trap 'echo failed > "$ROOT/$VARIANT/experiment_status"' ERR
export PYTHONPATH="$REPO/scripts/practice9:${PYTHONPATH:-}"
echo training > "$ROOT/$VARIANT/experiment_status"
bash scripts/practice9/train_resume_recovery.sh "$GPU" "$VARIANT" formal
cp "$ROOT/${VARIANT}_recovery/final_checkpoint" "$E/neck_only/final_checkpoint"
cp "$ROOT/${VARIANT}_recovery/source_commit" "$E/neck_only/source_commit"
cp "$ROOT/${VARIANT}_recovery/final_checkpoint" "$ROOT/${VARIANT}_formal/final_checkpoint"
echo trained > "$ROOT/${VARIANT}_formal/status"
echo evaluating > "$ROOT/$VARIANT/experiment_status"
bash "$E/run.sh" "$GPU" neck_only
bash "$E/front.sh" "$GPU" neck_only
bash "$E/diagnostic_front.sh" "$GPU" neck_only
/root/gpufree-data/conda_envs/env_isaaclab/bin/python "$E/finish.py"
echo completed > "$ROOT/$VARIANT/experiment_status"
