#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/pose_velocity_fit_v1"
C="$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
python -u scripts/practice9/calibrate_supermimic_branch_pair.py --parent "$C" --collection "$R/static_perturb_v1" --branches "$R/orthogonal_branches_v1" --arm candidate --pose-integration-fit-audit "$R/integration_consistency_v1/audit.json" --output "$S/fit" > "$S/fit.log" 2>&1
N="$S/fit/world_model_calibrated.pt"
python scripts/practice9/audit_supermimic_offline_calibration_integrity.py --artifact "$N" --source-checkpoint "$C" --collection "$R/static_perturb_v1" --output "$S/integrity.json" > "$S/integrity.log" 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$N" --output "$S/trajectory.json" > "$S/trajectory.log" 2>&1
for base in initial_h8 same_budget; do
 if [[ "$base" == initial_h8 ]]; then before="$R/standing_wm_calibration_v1/after.json"; else before="$R/standing_branch_pair_v1/candidate/after.json"; fi
 python scripts/practice9/compare_supermimic_calibration.py --before "$before" --after "$S/trajectory.json" --output "$S/$base.gate.json" --nonregression-only > "$S/$base.gate.log" 2>&1
done
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root "$R/standing_branch_pair_v1" --branches "$R/orthogonal_branches_v1" --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent "$C" --candidate-checkpoint "$N" --output "$S/branch_validation.json" > "$S/branch_validation.log" 2>&1
printf 'completed_pending_review\n' > "$S/status"
