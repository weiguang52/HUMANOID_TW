#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
STATE="$DATA/datasets/practice9/s2_squat_test"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}" P9_CUSTOM_ALLOW_UNSAFE_MOTIONS=1
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT
cd "$REPO"
CHECKPOINT=$(cat "$DATA/datasets/practice9/s2_control_fix/long_velocity/final_checkpoint")
RUN=$(dirname "$CHECKPOINT")
trap 'code=$?; if ((code)); then echo failed:$code > "$STATE/status"; fi' EXIT
echo evaluating > "$STATE/status"
for ID in 000890 001240; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/$ID.manifest.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 touch "$STATE/$ID.start"
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --video --video_length "$LENGTH" --seed 42 --disable_observation_noise --evaluation_output "$STATE/$ID.evaluation.json" --telemetry_output "$STATE/$ID.telemetry.npz" --viewer_follow_asset robot --viewer_follow_tau .5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$STATE/$ID.log" 2>&1
 VIDEO=$(find "$RUN/videos/play" -maxdepth 1 -name '*.mp4' -newer "$STATE/$ID.start" | head -1)
 test -s "$VIDEO"
 mv "$VIDEO" "$STATE/$ID.side.mp4"
 ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,nb_frames,r_frame_rate,width,height -of json "$STATE/$ID.side.mp4" > "$STATE/$ID.encoding.json"
 ffmpeg -nostdin -y -v error -i "$STATE/$ID.side.mp4" -vf 'fps=1/3,scale=480:-1,tile=3x2' -frames:v 1 "$STATE/$ID.preview.png"
done
echo completed > "$STATE/status"
