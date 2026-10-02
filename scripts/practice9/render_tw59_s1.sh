#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
STATE="$DATA/datasets/practice9/tw59_s1"
trap 'code=$?; if (( code != 0 )); then echo failed:$code > "$STATE/status"; fi' EXIT
export GPUFREE_DATA_ROOT="$DATA" ISAACLAB_PATH="$DATA/projects/IsaacLab"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/tw56_native_v1/manifest.json"
export P9_CUSTOM_GPU=1 P9_CUSTOM_NUM_ENVS=1024 P9_CUSTOM_MAX_ITERATIONS=10000 P9_CUSTOM_SEED=42
export P9_CUSTOM_RUN_NAME=tw59_s1_v2
export CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export TMPDIR="$DATA/tmp/practice9_custom" XDG_CACHE_HOME="$DATA/.cache/xdg"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
RUN=$(find "$REPO/logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d" -maxdepth 1 -type d -name '*_tw59_s1_v2' | sort | tail -1)
test -n "$RUN"
CHECKPOINT="$RUN/model_9999.pt"
test -s "$CHECKPOINT"
printf '%s\n' "$RUN" > "$STATE/run_dir"
printf '%s\n' "$CHECKPOINT" > "$STATE/checkpoint"
echo rendering > "$STATE/status"
python - "$STATE" <<'PY'
import json,sys
from pathlib import Path
state=Path(sys.argv[1]); original=json.loads((state/'training_manifest.json').read_text())
for motion in original['motions']:
    data=dict(original); data['motions']=[motion]
    (state/(motion['id']+'.manifest.json')).write_text(json.dumps(data,indent=2))
PY
mkdir -p "$STATE/videos"
for ID in ${TW59_VIDEO_IDS:-000801 006680 010407 010782}; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/$ID.manifest.json"
 VIDEO_LENGTH=$(python -c 'import json,sys; print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$STATE/$ID.manifest.json")
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_output "$STATE/$ID.evaluation.json" --video --video_length "$VIDEO_LENGTH" --viewer_follow_asset robot --viewer_eye 3 3 2 --viewer_lookat 0 0 0.7 > "$STATE/render_$ID.log" 2>&1
 VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' | head -1)
 test -s "$VIDEO"
 mv "$VIDEO" "$STATE/videos/$ID.mp4"
done
date -Iseconds > "$STATE/finished_at"
echo completed > "$STATE/status"
