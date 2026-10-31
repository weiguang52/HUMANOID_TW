#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/standing_branch_pair_v1"
C="$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
for arm in control candidate; do
 python -u scripts/practice9/calibrate_supermimic_branch_pair.py --parent "$C" --collection "$R/static_perturb_v1" --branches "$R/orthogonal_branches_v1" --arm "$arm" --output "$S/$arm/fit" > "$S/$arm.fit.log" 2>&1
 python scripts/practice9/audit_supermimic_offline_calibration_integrity.py --artifact "$S/$arm/fit/world_model_calibrated.pt" --source-checkpoint "$C" --collection "$R/static_perturb_v1" --output "$S/$arm/integrity.json" > "$S/$arm.integrity.log" 2>&1
 python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$S/$arm/fit/world_model_calibrated.pt" --output "$S/$arm/after.json" > "$S/$arm.audit.log" 2>&1
 python scripts/practice9/compare_supermimic_calibration.py --before "$R/standing_wm_calibration_v1/after.json" --after "$S/$arm/after.json" --output "$S/$arm/initial_h8_gate.json" --nonregression-only > "$S/$arm.gate.log" 2>&1
done
python scripts/practice9/compare_supermimic_calibration.py --before "$S/control/after.json" --after "$S/candidate/after.json" --output "$S/paired_trajectory_gate.json" --nonregression-only > "$S/paired_gate.log" 2>&1
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root "$S" --branches "$R/orthogonal_branches_v1" --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent "$C" > "$S/branch_validation.log" 2>&1
printf 'audited_pending_review\n' > "$S/status"
