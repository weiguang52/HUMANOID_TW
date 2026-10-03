#!/bin/bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
DATA=/root/gpufree-data
STATE=$DATA/datasets/practice9/tw91_slip_v1
OLD=$DATA/datasets/practice9/tw75_contact_v1
OUT=$STATE/control
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
trap 'echo failed > "$OUT/status"' ERR
exec 9>"$STATE/evaluation.lock"
flock -n -x 9
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE
unset P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT P9_CONTACT_PHASE_WEIGHT
echo evaluating > "$OUT/status"
for TRAIN_SEED in 123 42; do
 JOB=$OUT/seed$TRAIN_SEED
 CHECKPOINT=$(cat "$JOB/final_checkpoint")
 RUN=$(dirname "$CHECKPOINT")
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/evaluation_manifest.json"
 for EVAL_SEED in 42 123 2026; do
  [[ ! -f "$JOB/jobs$EVAL_SEED.json.done" ]] || continue
  python scripts/practice9/tw91_batch_jobs.py "$STATE/evaluation_manifest.json" "$JOB" "$EVAL_SEED"
  timeout --signal=TERM --kill-after=15s 1800s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --seed "$EVAL_SEED" --disable_observation_noise --evaluation_batch "$JOB/jobs$EVAL_SEED.json" > "$JOB/batch$EVAL_SEED.repair.log" 2>&1
  test -s "$JOB/jobs$EVAL_SEED.json.done"
 done
 echo metrics_completed > "$JOB/status"
 mkdir -p "$JOB/videos"
 while read -r ID; do
  if [[ -s "$JOB/videos/$ID.mp4" ]] && ffprobe -v error "$JOB/videos/$ID.mp4" >/dev/null 2>&1; then continue; fi
  export PRACTICE9_CUSTOM_MOTION_MANIFEST="$OLD/eval/$ID.json"
  timeout --signal=TERM --kill-after=15s 600s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$JOB/$ID.video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$JOB/$ID.video.repair.log" 2>&1
  VIDEO="$RUN/videos/play/rl-video-step-0.mp4"
  test -s "$VIDEO"
  ffprobe -v error "$VIDEO" > /dev/null
  mv "$VIDEO" "$JOB/videos/$ID.mp4"
 done < "$DATA/datasets/practice9/tw75_train_v1/eval_ids.txt"
 echo completed > "$JOB/status"
done
echo completed > "$OUT/status"
python scripts/practice9/summarize_tw91_slip.py
