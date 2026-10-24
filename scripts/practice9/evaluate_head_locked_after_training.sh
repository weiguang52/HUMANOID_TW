#!/bin/bash
set -eo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/head_locked_20261024
exec 8>"$STATE/evaluation_wait.lock"; flock -n 8
while true; do
 STATUS=$(cat "$STATE/status")
 [[ "$STATUS" != failed ]] || exit 1
 [[ "$STATUS" != trained ]] || break
 sleep 30
done
cp "$STATE/final_checkpoint" "$STATE/evaluation/neck_only/final_checkpoint"
cp "$STATE/source_commit" "$STATE/evaluation/neck_only/source_commit"
export PYTHONPATH="$PWD/scripts/practice9:${PYTHONPATH:-}"
bash "$STATE/evaluation/run.sh" 1 neck_only
