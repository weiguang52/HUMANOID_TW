#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
V="${1:?variant}"; GPU="${2:?gpu}"
STATE="$DATA/datasets/practice9/s2_joint_analysis"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_trials/$V/000801.manifest.json"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
trap 'code=$?; echo "$code" > "$STATE/${V}_side.exit"' EXIT
CHECKPOINT=$(cat "$DATA/datasets/practice9/s2_trials/$V/final_checkpoint")
RUN=$(dirname "$CHECKPOINT")
python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$STATE/${V}_side.evaluation.json" --viewer_follow_asset robot --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$STATE/${V}_side.log" 2>&1
VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' | head -1)
test -s "$VIDEO"
mv "$VIDEO" "$STATE/${V}_walk_side.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,nb_frames,r_frame_rate,width,height -of json "$STATE/${V}_walk_side.mp4" > "$STATE/${V}_side.encoding.json"
ffmpeg -v error -ss 5 -i "$STATE/${V}_walk_side.mp4" -frames:v 1 -y "$STATE/${V}_side_preview.png"
