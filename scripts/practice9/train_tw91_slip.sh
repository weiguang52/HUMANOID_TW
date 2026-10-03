#!/bin/bash
# Paired seeds; only feet_slide weight differs. All checkpoints retained.
set -eo pipefail
GPU=$1; VARIANT=$2; WEIGHT=$3; ITERATIONS=${4:-4000}
[[ "$VARIANT" == control || "$VARIANT" == slip ]]
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
DATA=/root/gpufree-data
STATE=$DATA/datasets/practice9/tw91_slip_v1
OLD=$DATA/datasets/practice9/tw75_contact_v1
OUT=$STATE/$VARIANT
mkdir -p "$OUT"
[[ ! -e "$OUT/status" ]] || { echo 'Existing run: refusing overwrite'; exit 1; }
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$OLD/splits/train.json"
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
INITIAL=$(cat "$OLD/contact/final_checkpoint")
FRAMES=$(python -c 'import json,sys;print(sum(x["frames"] for x in json.load(open(sys.argv[1]))["motions"]))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
git rev-parse HEAD > "$OUT/source_commit"
printf '%s\n' "$INITIAL" > "$OUT/initial_checkpoint"
printf '%s\n' "$WEIGHT" > "$OUT/feet_slide_weight"
for TRAIN_SEED in 42 123; do
 JOB=$OUT/seed$TRAIN_SEED
 mkdir -p "$JOB"
 echo "training_seed$TRAIN_SEED" > "$OUT/status"
 echo training > "$JOB/status"
 RUN_NAME=tw91_${VARIANT}_s${TRAIN_SEED}
 python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1024 --max_iterations "$ITERATIONS" --seed "$TRAIN_SEED" --run_name "$RUN_NAME" --logger tensorboard --resume --load_run "$(basename "$(dirname "$INITIAL")")" --checkpoint "$(basename "$INITIAL")" "env.rewards.feet_slide.weight=$WEIGHT" env.rewards.action_rate_l2.weight=-0.2 env.rewards.joint_acc.weight=-2e-6 env.rewards.motion_joint_vel_cost.weight=-0.2 env.commands.motion.adaptive_uniform_ratio=1.0 "env.commands.motion.max_motion_frames=$FRAMES" agent.save_interval=500 > "$JOB/train.log" 2>&1
 RUN=$(find logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
 CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
 test -s "$CHECKPOINT"
 realpath "$CHECKPOINT" > "$JOB/final_checkpoint"
 echo trained > "$JOB/status"
done
echo done > "$OUT/training.done"
echo waiting_for_serial_evaluation > "$OUT/status"
OTHER=control; [[ "$VARIANT" != control ]] || OTHER=slip
while [[ ! -f "$STATE/$OTHER/training.done" ]]; do
 [[ "$(cat "$STATE/$OTHER/status" 2>/dev/null || true)" != failed ]] || break
 sleep 20
done
exec 9>"$STATE/evaluation.lock"
flock -x 9
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE
unset P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT P9_CONTACT_PHASE_WEIGHT
echo evaluating > "$OUT/status"
for TRAIN_SEED in 42 123; do
 JOB=$OUT/seed$TRAIN_SEED
 CHECKPOINT=$(cat "$JOB/final_checkpoint")
 RUN=$(dirname "$CHECKPOINT")
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/evaluation_manifest.json"
 for EVAL_SEED in 42 123 2026; do
  python scripts/practice9/tw91_batch_jobs.py "$STATE/evaluation_manifest.json" "$JOB" "$EVAL_SEED"
  timeout --signal=TERM --kill-after=15s 1800s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --seed "$EVAL_SEED" --disable_observation_noise --evaluation_batch "$JOB/jobs$EVAL_SEED.json" > "$JOB/batch$EVAL_SEED.log" 2>&1
  test -s "$JOB/jobs$EVAL_SEED.json.done"
 done
 echo metrics_completed > "$JOB/status"
 mkdir -p "$JOB/videos"
 while read -r ID; do
  export PRACTICE9_CUSTOM_MOTION_MANIFEST="$OLD/eval/$ID.json"
  timeout --signal=TERM --kill-after=15s 600s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$JOB/$ID.video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$JOB/$ID.video.log" 2>&1
  VIDEO="$RUN/videos/play/rl-video-step-0.mp4"
  test -s "$VIDEO"
  ffprobe -v error "$VIDEO" > /dev/null
  mv "$VIDEO" "$JOB/videos/$ID.mp4"
 done < "$DATA/datasets/practice9/tw75_train_v1/eval_ids.txt"
 echo completed > "$JOB/status"
done
echo completed > "$OUT/status"
python scripts/practice9/summarize_tw91_slip.py
