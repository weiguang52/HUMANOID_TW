#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
GPU="${1:?GPU}"; VARIANT="${2:?variant}"
OUT="$DATA/datasets/practice9/s2_trials/$VARIANT"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
unset P9_CUSTOM_ALLOW_UNSAFE_MOTIONS
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
trap 'code=$?; if ((code)); then echo failed:$code > "$OUT/validation.status"; fi' EXIT
echo waiting_for_training > "$OUT/validation.status"
while true; do
 STATUS=$(cat "$OUT/status")
 case "$STATUS" in trained) break;; failed:*) exit 1;; training) sleep 20;; *) exit 2;; esac
done
RUN=$(find "$REPO/logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d" -maxdepth 1 -type d -name "*_s2_trial_$VARIANT" | sort | tail -1)
test -n "$RUN"
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"
printf '%s\n' "$CHECKPOINT" > "$OUT/final_checkpoint"
mkdir -p "$OUT/videos"
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.load(open(p/'training_manifest.json'))
for r in m['motions']:
 x=dict(m);x['motions']=[r];(p/(r['id']+'.manifest.json')).write_text(json.dumps(x,indent=2)+'\n')
PY
echo evaluating > "$OUT/validation.status"
for ID in 000801 006680 010407 010782 s2_stand s2_weight_shift s2_wave_000113; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$OUT/$ID.manifest.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length "$LENGTH" --seed 42 --disable_observation_noise --evaluation_output "$OUT/$ID.evaluation.json" --telemetry_output "$OUT/$ID.telemetry.npz" --viewer_follow_asset robot --viewer_eye 3 3 2 --viewer_lookat 0 0 0.7 > "$OUT/$ID.evaluation.log" 2>&1
 VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' | head -1)
 test -s "$VIDEO"
 mv "$VIDEO" "$OUT/videos/$ID.mp4"
 ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,nb_frames,r_frame_rate -of json "$OUT/videos/$ID.mp4" > "$OUT/videos/$ID.encoding.json"
done
echo completed > "$OUT/validation.status"
