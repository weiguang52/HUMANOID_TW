#!/bin/bash
set -eo pipefail
GPU=${1:?}; MODEL=${2:?}
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 P9_EXPRESSION_TELEMETRY=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
STATE=/root/gpufree-data/datasets/practice9/tw198_v1
export PRACTICE9_CUSTOM_MOTION_MANIFEST=$STATE/manifest.json
CHECKPOINT=$(cat /root/gpufree-data/datasets/practice9/tw195_v1/$MODEL/final_checkpoint)
trap 'echo failed > "$STATE/$MODEL/status"' ERR
for RATE in 200hz 400hz; do
 DT=.005; CAMERA=()
 if [[ "$RATE" == 400hz ]]; then DT=.0025; else CAMERA=(--enable_cameras); fi
 if [[ "$RATE" == 200hz ]]; then exec 9>"$STATE/video.lock"; flock -x 9; fi
 echo "$RATE" > "$STATE/$MODEL/status"
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-PathRate-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --seed 42 --standing_diagnostic --diagnostic_physics_dt "$DT" --evaluation_batch "$STATE/$MODEL/$RATE/jobs.json" "${CAMERA[@]}" > "$STATE/$MODEL/$RATE/run.log" 2>&1
 if [[ "$RATE" == 200hz ]]; then flock -u 9; fi
done
echo completed > "$STATE/$MODEL/status"
