#!/usr/bin/env bash
# Fixed-budget diagnostic; no policy promotion or checkpoint selection.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export MUJOCO_GL=egl OMP_NUM_THREADS=4
root=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031
state="$root/longterm_diagnostics/real_ppo_standing_v1"
parent=/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt
manifest="$root/static_probe/manifest.json"
mkdir -p "$state"
# Exclusive run ownership; reruns must explicitly resume the latest completed update.
exec 9>"$state/runner.lock"
flock -n 9 || { echo 'Diagnostic already running' >&2; exit 1; }
trap 'code=$?; if (( code )); then echo "failed_exit_$code" > "$state/experiment_status"; fi' EXIT
export TW_SUPERMIMIC_MJCF="$root/robot/tw30.xml"
if [[ ! -f "$state/checkpoint_final.pt" ]]; then
  args=()
  if [[ -f "$state/metrics.jsonl" ]]; then
    latest=$(find "$state" -maxdepth 1 -name 'checkpoint_[0-9][0-9][0-9][0-9].pt' | sort | tail -n 1)
    [[ -n "$latest" ]] || { echo 'Existing incomplete run has no completed checkpoint; inspect it before recovery' >&2; exit 1; }
    args+=(--resume "$latest")
  fi
  echo training > "$state/experiment_status"
  python -u scripts/practice9/train_supermimic_real_ppo.py --parent "$parent" --manifest "$manifest" --output "$state" --num-envs 256 --updates 1000 "${args[@]}"
fi
export TW_SUPERMIMIC_MJCF="$root/robot/tw30_colored.xml"
if [[ ! -f "$state/evaluation/results.json" ]]; then
  # Preserve incomplete evaluation rather than mixing episodes from different attempts.
  [[ ! -d "$state/evaluation" ]] || { echo 'Inspect incomplete evaluation before retry' >&2; exit 1; }
  echo evaluating > "$state/experiment_status"
  python -u scripts/practice9/evaluate_supermimic_tw.py --checkpoint "$state/checkpoint_final.pt" --manifest "$manifest" --output "$state/evaluation" --max-steps 1000 --seed 42 --agents trained zero --video --native-contacts
fi
if [[ ! -f "$state/evaluation/episode_media_integrity.json" ]]; then
  python scripts/practice9/verify_supermimic_episode_media.py --evaluation "$state/evaluation"
fi
echo evaluated_pending_visual_and_standing_review > "$state/experiment_status"
