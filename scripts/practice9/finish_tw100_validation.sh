#!/bin/bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/tw100_expression_v1
VAL=$STATE/validation
trap 'echo failed > "$VAL/delivery.status"' ERR
echo waiting_for_replays > "$VAL/delivery.status"
while true; do
 A=$(cat "$VAL/expression/status"); B=$(cat "$VAL/control/status")
 [[ "$A" != failed && "$B" != failed ]]
 [[ "$A" != completed || "$B" != completed ]] || break
 sleep 15
done
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=0.5
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
# 000346 fails after28s in one quantitative replay; a20s excerpt would conceal it.
for V in expression control; do
 OUT=$VAL/$V
 echo "full_video_$V" > "$VAL/delivery.status"
 TASK=Unitree-Custom-Humanoid-30dof-Expression-HumanML3D
 [[ "$V" != control ]] || TASK=Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D
 CHECKPOINT=$(cat "$STATE/$V/final_checkpoint")
 export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/tw75_contact_v1/eval/000346.json
 timeout --signal=TERM --kill-after=15s 1200s python scripts/rsl_rl/play.py --headless --task "$TASK" --num_envs 1 --checkpoint "$CHECKPOINT" --seed 42 --disable_observation_noise --evaluation_upper_body_termination --evaluation_motion_id 0 --video --rendering_mode performance --video_length 1788 --evaluation_output "$OUT/000346.full_video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$OUT/000346.full_video.log" 2>&1
 VIDEO="$(dirname "$CHECKPOINT")/videos/play/rl-video-step-0.mp4"
 test -s "$VIDEO"
 ffprobe -v error "$VIDEO" >/dev/null
 mv "$OUT/videos/000346.mp4" "$OUT/000346_excerpt.mp4"
 mv "$VIDEO" "$OUT/videos/000346.mp4"
done
python scripts/practice9/summarize_tw100_validation.py > "$VAL/summary.log"
python scripts/practice9/package_tw100_validation.py > "$VAL/package.log"
echo artifacts_ready > "$VAL/delivery.status"
