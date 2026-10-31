#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/policy_anchor_smoke_v1"
mkdir "$S"
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
python -u scripts/practice9/train_supermimic_standing_calibration_pair.py \
 --pilot-updates 4 --num-envs 8 --branch calibrated_retained --zero-readout --output "$S/calibrated_retained" \
 --anchor-collection "$R/static_perturb_v1" \
 --policy-anchor-collection "$R/static_perturb_v1" \
 --run-root /root/gpufree-data/twrl_runs/supermimic_tw_20261031/policy_anchor_smoke_v1 \
 --checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt \
 --dynamics-checkpoint "$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt" \
 --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json > "$S/train.log" 2>&1
printf 'smoke_trained_pending_checks\n' > "$S/status"
