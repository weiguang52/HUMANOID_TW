#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/learned_policy_native_contacts_v1"
mkdir "$S"
printf 'waiting_for_original_evaluation\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
# Wait only for the confirmed original evaluator; never restart it.
WAIT_PID=${1:-0}
if [[ "$WAIT_PID" != 0 ]]; then
 while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 5; done
fi
test -f "$R/standing_policy_anchor_v1/evaluation/results.json"
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
printf 'running\n' > "$S/status"
for case in standing_retention_zero_readout_v1 standing_policy_anchor_v1; do
 python -u scripts/practice9/evaluate_supermimic_tw.py --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json --checkpoint "/root/gpufree-data/twrl_runs/supermimic_tw_20261031/$case/calibrated_retained/checkpoint_final.pt" --output "$S/$case" --max-steps 1000 --agents trained zero --native-contacts > "$S/$case.log" 2>&1
done
printf 'collected_pending_audit\n' > "$S/status"
