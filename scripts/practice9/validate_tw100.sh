#!/bin/bash
set -eo pipefail
GPU=${1:?}; VARIANT=${2:?}
[[ "$VARIANT" == expression || "$VARIANT" == control ]]
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/tw100_expression_v1
OUT=$STATE/validation/$VARIANT
exec 8>"$OUT/run.lock"
flock -n 8
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=0.5
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/validation/manifest.json"
CHECKPOINT=$(cat "$STATE/$VARIANT/final_checkpoint")
TASK=Unitree-Custom-Humanoid-30dof-Expression-HumanML3D
[[ "$VARIANT" != control ]] || TASK=Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D
for SEED in 42 123 2026; do
 [[ ! -f "$OUT/jobs$SEED.json.done" ]] || continue
 echo "metrics_seed$SEED" > "$OUT/status"
 timeout --signal=TERM --kill-after=15s 1800s python scripts/rsl_rl/play.py --headless --task "$TASK" --num_envs 1 --checkpoint "$CHECKPOINT" --seed "$SEED" --disable_observation_noise --evaluation_upper_body_termination --evaluation_batch "$OUT/jobs$SEED.json" > "$OUT/batch$SEED.log" 2>&1
 test -s "$OUT/jobs$SEED.json.done"
done
echo waiting_for_video_lock > "$OUT/status"
exec 9>"$STATE/validation/video.lock"
flock -x 9
mkdir -p "$OUT/videos"
for ID in 000016 000124 000346; do
 if [[ -s "$OUT/videos/$ID.mp4" ]] && ffprobe -v error "$OUT/videos/$ID.mp4" >/dev/null 2>&1; then continue; fi
 echo "video_$ID" > "$OUT/status"
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="/root/gpufree-data/datasets/practice9/tw75_contact_v1/eval/$ID.json"
 timeout --signal=TERM --kill-after=15s 900s python scripts/rsl_rl/play.py --headless --task "$TASK" --num_envs 1 --checkpoint "$CHECKPOINT" --seed 42 --disable_observation_noise --evaluation_upper_body_termination --evaluation_motion_id 0 --video --rendering_mode performance --video_length 1000 --evaluation_output "$OUT/$ID.video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$OUT/$ID.video.log" 2>&1
 VIDEO="$(dirname "$CHECKPOINT")/videos/play/rl-video-step-0.mp4"
 test -s "$VIDEO"
 ffprobe -v error "$VIDEO" >/dev/null
 mv "$VIDEO" "$OUT/videos/$ID.mp4"
done
echo completed > "$OUT/status"
