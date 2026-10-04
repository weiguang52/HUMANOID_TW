#!/bin/bash
set -eo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
S=/root/gpufree-data/datasets/practice9/tw123_v1
trap 'echo failed > "$S/pipeline.status"' ERR
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
echo waiting_data > "$S/pipeline.status"
while :; do
 A=$(cat "$S/processed/prepare_0.status" 2>/dev/null || true)
 B=$(cat "$S/processed/prepare_1.status" 2>/dev/null || true)
 [[ "$A" != failed && "$B" != failed ]]
 if [[ "$A" == completed && "$B" == completed && -f "$S/pilot_review.json" ]]; then break; fi
 sleep 15
done
python scripts/practice9/finalize_tw123_data.py > "$S/finalize.log" 2>&1
# Start only after explicit source commit was pushed, never use old policies.
while [[ ! -f "$S/source_pushed" ]]; do sleep 15; done
test "$(cat "$S/source_pushed")" = "$(git rev-parse HEAD)"
echo smoke_training > "$S/pipeline.status"
PIDS=()
for SPEC in '0 bounded' '1 balanced'; do
 read -r GPU VARIANT <<< "$SPEC"
 if [[ "$(cat "$S/smoke_$VARIANT/status" 2>/dev/null || true)" == trained ]]; then continue; fi
 bash scripts/practice9/train_tw123_mixed.sh "$GPU" "$VARIANT" 3 smoke > "$S/smoke_$VARIANT.log" 2>&1 &
 PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
python scripts/practice9/audit_tw123_start.py > "$S/start_audit.json"
echo training > "$S/pipeline.status"
tmux new-session -d -s tw123_mixed_bounded "cd $PWD && bash scripts/practice9/train_tw123_mixed.sh 0 bounded 12000 > $S/bounded_supervisor.log 2>&1"
tmux new-session -d -s tw123_mixed_balanced "cd $PWD && bash scripts/practice9/train_tw123_mixed.sh 1 balanced 12000 > $S/balanced_supervisor.log 2>&1"
