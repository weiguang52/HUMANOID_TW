#!/bin/bash
set -eo pipefail
GPU=$1; SEED=$2
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
DATA=/root/gpufree-data
STATE=$DATA/datasets/practice9/tw75_train_v1
OUT=$STATE/seed$SEED
mkdir -p "$OUT"
trap 'echo failed > "$OUT/status"' ERR
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}" LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
echo waiting_for_data > "$OUT/status"
while :; do
 A=$(cat "$STATE/prepare_0.status" 2>/dev/null || true); B=$(cat "$STATE/prepare_1.status" 2>/dev/null || true)
 if [[ "$A" == failed || "$B" == failed ]]; then echo preparation_failed > "$OUT/status"; exit 1; fi
 if [[ "$A" == completed && "$B" == completed ]]; then break; fi
 sleep 30
done
if [[ "$GPU" == 0 ]]; then
 python scripts/practice9/build_tw75_manifests.py --selection "$DATA/datasets/practice9/tw75_dataset_v1/selection.json" --fk-manifests "$STATE"/batches/*/manifest.json --output "$STATE/splits"
 python - "$STATE" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);d=json.load(open(r/'splits/val.json'));chosen=[]
for category in ['walking','standing_upper']:
 chosen.extend([x for x in d['motions'] if x['category']==category][:3])
if len(chosen)!=6:raise ValueError('Insufficient held-out validation coverage')
(r/'eval').mkdir(exist_ok=True)
for x in chosen:
 p=dict(d);p['motions']=[x];(r/'eval'/f'{x["id"]}.json').write_text(json.dumps(p))
(r/'eval_ids.txt').write_text(''.join(x['id']+'\n' for x in chosen))
(r/'splits/ready').write_text('ready')
PY
fi
while [[ ! -f "$STATE/splits/ready" ]]; do
 if [[ "$(cat "$STATE/seed42/status" 2>/dev/null || true)" == failed ]]; then echo manifest_build_failed > "$OUT/status"; exit 1; fi
 sleep 15
done
export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/splits/train.json"
FRAMES=$(python -c 'import json,sys;print(sum(x["frames"] for x in json.load(open(sys.argv[1]))["motions"]))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5 P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1
CHECKPOINT=$(cat "$DATA/datasets/practice9/s2_control_fix/long_velocity/final_checkpoint")
git rev-parse HEAD > "$OUT/source_commit"
cp "$PRACTICE9_CUSTOM_MOTION_MANIFEST" "$OUT/training_manifest.json"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1024 --max_iterations 20000 --seed "$SEED" --run_name "tw75_full_s$SEED" --logger tensorboard --resume --load_run "$(basename "$(dirname "$CHECKPOINT")")" --checkpoint "$(basename "$CHECKPOINT")" env.rewards.action_rate_l2.weight=-0.2 env.rewards.joint_acc.weight=-2e-6 env.rewards.motion_joint_vel_cost.weight=-0.2 env.commands.motion.adaptive_uniform_ratio=1.0 "env.commands.motion.max_motion_frames=$FRAMES" agent.save_interval=500 > "$OUT/train.log" 2>&1
RUN=$(find logs/rsl_rl/unitree_custom_humanoid_30dof_mimic_humanml3d -maxdepth 1 -type d -name "*_tw75_full_s$SEED" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
unset P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY P9_TARGET_VELOCITY_SCALE P9_PD_STIFFNESS_MULT P9_PD_DAMPING_MULT
echo evaluating > "$OUT/status"
mkdir -p "$OUT/videos"
while read -r ID; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/eval/$ID.json"
 LENGTH=$(python -c 'import json,sys;print(max(1000,json.load(open(sys.argv[1]))["motions"][0]["frames"]+1))' "$PRACTICE9_CUSTOM_MOTION_MANIFEST")
 for EVAL_SEED in 42 123 2026; do
  python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps "$LENGTH" --seed "$EVAL_SEED" --disable_observation_noise --evaluation_output "$OUT/$ID.seed$EVAL_SEED.json" --telemetry_output "$OUT/$ID.seed$EVAL_SEED.npz" > "$OUT/$ID.seed$EVAL_SEED.log" 2>&1
 done
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --video --video_length 1000 --seed 42 --disable_observation_noise --evaluation_output "$OUT/$ID.video.json" --viewer_follow_asset robot --viewer_follow_tau 0.5 --viewer_eye 0 -1.2 0.15 --viewer_lookat 0 0 -0.03 > "$OUT/$ID.video.log" 2>&1
 VIDEO=$(find "$RUN/videos/play" -name '*.mp4' | head -1)
 test -s "$VIDEO"; mv "$VIDEO" "$OUT/videos/$ID.mp4"
done < "$STATE/eval_ids.txt"
echo completed > "$OUT/status"
