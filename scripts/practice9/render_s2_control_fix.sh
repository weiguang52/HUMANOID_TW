#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
V="${1:?variant}"; GPU="${2:?GPU}"
STATE="$DATA/datasets/practice9/s2_control_fix/$V"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
CHECKPOINT=$(cat "$STATE/final_checkpoint");RUN=$(dirname "$CHECKPOINT")
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$DATA/datasets/practice9/s2_trials/smooth/000801.manifest.json"
python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$STATE/side.evaluation.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$STATE/side.log" 2>&1
VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' | head -1);test -s "$VIDEO"
mv "$VIDEO" "$STATE/walk_side_stable_camera.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,nb_frames,r_frame_rate,width,height -of json "$STATE/walk_side_stable_camera.mp4" > "$STATE/side.encoding.json"
ffmpeg -v error -ss 5 -i "$STATE/walk_side_stable_camera.mp4" -frames:v 1 -y "$STATE/side.preview.png"
