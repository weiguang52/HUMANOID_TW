#!/bin/bash
set -eo pipefail
ROOT=/root/gpufree-data/datasets/practice9
BASE=$ROOT/velocity_balance_20261029
SEED=$1; GPU=$2
if [[ "$SEED" = 42 ]]; then S=$BASE; else S=$ROOT/velocity_balance_seed123_20261029; fi
D=$S/videos
exec > "$D/worker.log" 2>&1
exec 8>"$D/worker.lock"; flock -n 8
trap 'echo failed > "$D/status"' ERR
PY=/root/gpufree-data/conda_envs/env_isaaclab/bin/python
"$PY" -c 'import json,sys;assert json.load(open(sys.argv[1]))["both_training_seeds_numeric_pass"]' "$BASE/two_training_seed_report.json"
CKPT=$(cat "$S/checkpoint")
EXPECTED=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["sha256"])' "$S/checkpoint_freeze.json")
test "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$EXPECTED"
for GROUP in validation independent; do
 for VIEW in front side; do
  echo "${GROUP}_${VIEW}" > "$D/status"
  bash "$D/$GROUP/run.sh" "$VIEW" "$GPU"
 done
done
for VIEW in front side; do
 echo "retention_${VIEW}" > "$D/status"
 bash "$BASE/run_retention_video.sh" "$S" "$VIEW" "$GPU"
done
echo auditing > "$D/status"
"$PY" "$D/audit_videos.py" > "$D/video_audit.log"
"$PY" -c 'import json,sys;x=json.load(open(sys.argv[1]));assert x["complete"] and x["all_observed_pass"]' "$D/video_audit.json"
"$PY" "$D/audit_source_videos.py" > "$D/source_video_audit.log"
"$PY" "$D/qa_sheets.py" > "$D/qa_sheets.log"
"$PY" "$BASE/audit_retention_videos.py" "$S" > "$D/retention_video_audit.log"
echo rendered_pending_startup_review > "$D/status"
