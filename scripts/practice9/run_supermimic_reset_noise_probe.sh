#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
export MUJOCO_GL=egl
S=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/reset_noise_probe_v2
python -u scripts/practice9/collect_supermimic_reset_noise_probe.py \
 --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json \
 --training-config /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_retention_zero_readout_v1/calibrated_retained/config.json \
 --output "$S"
