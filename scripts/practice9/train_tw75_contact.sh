#!/bin/bash
# Same seed/data/controller/checkpoint; only contact-phase weight differs.
set -eo pipefail
GPU=$1; VARIANT=$2; WEIGHT=$3; ITERATIONS=${4:-4000}
[[ "$VARIANT" == control || "$VARIANT" == contact ]]
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
DATA=/root/gpufree-data
STATE=$DATA/datasets/practice9/tw75_contact_v1
OUT=$STATE/$VARIANT
mkdir -p "$OUT"
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/splits/train.json"
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1
export P9_CONTACT_PHASE_WEIGHT=$WEIGHT
CHECKPOINT=$(cat "$DATA/datasets/practice9/tw75_train_v1/seed123/final_checkpoint")
FRAMES=$(python -c 'import json,sys;print(sum(x["frames"] for x in json.load(open(sys.argv[1]))["motions"]))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
git rev-parse HEAD > "$OUT/source_commit"
printf '%s\n' "$CHECKPOINT" > "$OUT/initial_checkpoint"
printf '%s\n' "$WEIGHT" > "$OUT/contact_weight"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1024 --max_iterations "$ITERATIONS" --seed 42 --run_name "tw75_contact_$VARIANT" --logger tensorboard --resume --load_run "$(basename "$(dirname "$CHECKPOINT")")" --checkpoint "$(basename "$CHECKPOINT")" env.rewards.action_rate_l2.weight=-0.2 env.rewards.joint_acc.weight=-2e-6 env.rewards.motion_joint_vel_cost.weight=-0.2 env.commands.motion.adaptive_uniform_ratio=1.0 "env.commands.motion.max_motion_frames=$FRAMES" agent.save_interval=500 > "$OUT/train.log" 2>&1
RUN=$(find logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d -maxdepth 1 -type d -name "*_tw75_contact_$VARIANT" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo done > "$OUT/training.done"
echo waiting_for_serial_evaluation > "$OUT/status"
OTHER=control; [[ "$VARIANT" != control ]] || OTHER=contact
while [[ ! -f "$STATE/$OTHER/training.done" ]]; do
 [[ "$(cat "$STATE/$OTHER/status" 2>/dev/null || true)" != failed ]] || break
 sleep 20
done
# Serialize all Isaac replays/renderers after both training jobs finish.
exec 9>"$STATE/evaluation.lock"
flock -x 9
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE
unset P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT P9_CONTACT_PHASE_WEIGHT
echo evaluating > "$OUT/status"
mkdir -p "$OUT/videos"
while read -r ID; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/eval/$ID.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 for EVAL_SEED in 42 123 2026; do
  timeout --signal=TERM --kill-after=15s 600s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed "$EVAL_SEED" --disable_observation_noise --evaluation_output "$OUT/$ID.seed$EVAL_SEED.json" --telemetry_output "$OUT/$ID.seed$EVAL_SEED.npz" > "$OUT/$ID.seed$EVAL_SEED.log" 2>&1
 done
 timeout --signal=TERM --kill-after=15s 600s python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$OUT/$ID.video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$OUT/$ID.video.log" 2>&1
 VIDEO="$RUN/videos/play/rl-video-step-0.mp4"
 test -s "$VIDEO"
 ffprobe -v error "$VIDEO" > /dev/null
 mv "$VIDEO" "$OUT/videos/$ID.mp4"
done < "$DATA/datasets/practice9/tw75_train_v1/eval_ids.txt"
echo completed > "$OUT/status"

python scripts/practice9/summarize_tw75_contact.py
