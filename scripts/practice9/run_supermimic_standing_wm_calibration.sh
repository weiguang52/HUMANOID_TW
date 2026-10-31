#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/standing_wm_calibration_v1"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
C=/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt
python -u scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$C" --output "$S/before.json" > "$S/before.log" 2>&1
python -u scripts/practice9/calibrate_supermimic_world_model.py --collection "$R/static_perturb_v1" --checkpoint "$C" --output "$S/fit" > "$S/fit.log" 2>&1
python -u scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$S/fit/world_model_calibrated.pt" --output "$S/after.json" > "$S/after.log" 2>&1
python scripts/practice9/compare_supermimic_calibration.py --before "$S/before.json" --after "$S/after.json" --output "$S/gate.json" > "$S/gate.log" 2>&1
printf 'completed_pending_review\n' > "$S/status"
