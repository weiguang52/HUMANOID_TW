#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/pose_integration_diagnostic_v1"
C="$R/standing_branch_pair_v1/candidate/fit/world_model_calibrated.pt"
A="$R/integration_consistency_v1/audit.json"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$C" --output "$S/trajectory.json" --pose-integration-fit-audit "$A" > "$S/trajectory.log" 2>&1
for source in initial_h8 control independent_candidate; do
 case "$source" in
 initial_h8) before="$R/standing_wm_calibration_v1/after.json";;
 control) before="$R/standing_branch_pair_v1/control/after.json";;
 independent_candidate) before="$R/standing_branch_pair_v1/candidate/after.json";;
 esac
 python scripts/practice9/compare_supermimic_calibration.py --before "$before" --after "$S/trajectory.json" --output "$S/$source.gate.json" --nonregression-only > "$S/$source.gate.log" 2>&1
done
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root "$R/standing_branch_pair_v1" --branches "$R/orthogonal_branches_v1" --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent "$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt" --pose-integration-fit-audit "$A" --output "$S/branch_validation.json" > "$S/branch_validation.log" 2>&1
printf 'completed_pending_review\n' > "$S/status"
