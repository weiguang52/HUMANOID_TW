#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/standing_retention_zero_readout_v1"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
python -u scripts/practice9/train_supermimic_standing_calibration_pair.py \
 --branch calibrated_retained --zero-readout --output "$S/calibrated_retained" \
 --anchor-collection "$R/static_perturb_v1" \
 --run-root /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_retention_zero_readout_v1 \
 --checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt \
 --dynamics-checkpoint "$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt" \
 --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json > "$S/train.log" 2>&1
C=/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_retention_zero_readout_v1/calibrated_retained/checkpoint_final.pt
printf 'model_audit\n' > "$S/status"
python -u scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$C" --output "$S/model_audit.json" > "$S/model_audit.log" 2>&1
python scripts/practice9/compare_supermimic_calibration.py --before "$R/standing_wm_calibration_v1/after.json" --after "$S/model_audit.json" --output "$S/retention_gate.json" --nonregression-only > "$S/retention_gate.log" 2>&1
python scripts/practice9/compare_supermimic_calibration.py --before "$R/standing_retention_v1/model_audit.json" --after "$S/model_audit.json" --output "$S/model_nonregression_gate.json" --nonregression-only > "$S/model_nonregression_gate.log" 2>&1
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml
printf 'evaluating\n' > "$S/status"
python -u scripts/practice9/evaluate_supermimic_tw.py \
 --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json \
 --checkpoint "$C" --output "$S/evaluation" --max-steps 1000 --agents trained zero --video > "$S/evaluation.log" 2>&1
printf 'evaluated_pending_review\n' > "$S/status"
