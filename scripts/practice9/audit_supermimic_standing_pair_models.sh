#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
for B in control calibrated; do
 python -u scripts/practice9/audit_supermimic_calibration_model.py \
 --collection "$R/static_perturb_v1" \
 --checkpoint "/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_calibration_pair_v1/$B/checkpoint_final.pt" \
 --output "$R/standing_calibration_pair_v1/${B}_static_model_audit.json" > "$R/standing_calibration_pair_v1/${B}_static_model_audit.log" 2>&1
done
