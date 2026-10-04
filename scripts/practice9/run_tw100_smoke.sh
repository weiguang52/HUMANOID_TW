#!/bin/bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
OUT=/root/gpufree-data/datasets/practice9/tw100_expression_v1
trap 'echo failed > "$OUT/smoke.status"' ERR
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/tw100_expression_v1/standing_train.json
FRAMES=$(python -c 'import json,os; print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
echo running > "$OUT/smoke.status"
python scripts/rsl_rl/train.py --headless --task Unitree-Custom-Humanoid-30dof-Expression-HumanML3D --num_envs 32 --max_iterations 2 --seed 42 --run_name tw100_smoke --logger tensorboard "env.commands.motion.max_motion_frames=$FRAMES" > "$OUT/smoke.log" 2>&1
echo completed > "$OUT/smoke.status"
