#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml
S=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_retention_v1/head_zero_diagnostic
mkdir "$S"
printf 'evaluating\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
python -u scripts/practice9/evaluate_supermimic_tw.py \
 --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json \
 --checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_retention_v1/calibrated_retained/checkpoint_final.pt \
 --output "$S/evaluation" --max-steps 1000 --agents trained trained_head_zero --video > "$S/evaluation.log" 2>&1
printf 'evaluated_pending_review\n' > "$S/status"
