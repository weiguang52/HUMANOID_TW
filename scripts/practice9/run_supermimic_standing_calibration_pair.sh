#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/standing_calibration_pair_v1"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
for B in control calibrated; do
 export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
 python -u scripts/practice9/train_supermimic_standing_calibration_pair.py \
  --branch "$B" --output "$S/$B" \
  --run-root /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_calibration_pair_v1 \
  --checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt \
  --dynamics-checkpoint "$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt" \
  --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json > "$S/$B.train.log" 2>&1
 export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml
 printf 'evaluating\n' > "$S/$B/status"
 python -u scripts/practice9/evaluate_supermimic_tw.py \
  --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json \
  --checkpoint "/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_calibration_pair_v1/$B/checkpoint_final.pt" \
  --output "$S/$B/evaluation" --max-steps 1000 --agents trained zero --video > "$S/$B.evaluation.log" 2>&1
 printf 'evaluated_pending_review\n' > "$S/$B/status"
done
printf 'completed_pending_review\n' > "$S/status"
