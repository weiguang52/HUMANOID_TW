#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
STATE="$DATA/datasets/practice9/nonwalk_s1_v2/diagnostic"
CHECKPOINT=$(cat "$DATA/datasets/practice9/tw59_s1/checkpoint")
RUN=$(dirname "$CHECKPOINT")
trap 'code=$?; if (( code != 0 )); then echo failed:$code > "$STATE/status"; fi' EXIT
export CUDA_VISIBLE_DEVICES=1 ISAACLAB_PATH="$DATA/projects/IsaacLab"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
# Isolated diagnostic replay: rejected references never become training data.
export P9_CUSTOM_ALLOW_UNSAFE_MOTIONS=1
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
python - "$STATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads((p/'manifest.json').read_text())
for m in d['motions']:
 x=dict(d);x['motions']=[m];(p/(m['id']+'.manifest.json')).write_text(json.dumps(x,indent=2))
PY
mkdir -p "$STATE/videos"
echo rendering > "$STATE/status"
for ID in 000113 000672 000433 000653; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/$ID.manifest.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$STATE/$ID.manifest.json")
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_output "$STATE/$ID.evaluation.json" --video --video_length "$LENGTH" --viewer_follow_asset robot --viewer_eye 1.2 1.2 0.8 --viewer_lookat 0 0 0.25 > "$STATE/render_$ID.log" 2>&1
 VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' | head -1)
 test -s "$VIDEO"
 mv "$VIDEO" "$STATE/videos/$ID.mp4"
done
echo completed > "$STATE/status"
