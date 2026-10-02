#!/bin/bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export MUJOCO_GL=egl
ROOT=/root/gpufree-data/datasets/practice9/tw75_dataset_v1
trap 'echo failed > "$ROOT/pairs.status"' ERR
echo rendering > "$ROOT/pairs.status"
python scripts/practice9/render_tw75_pairs.py --output "$ROOT/paired_visualization"
echo completed > "$ROOT/pairs.status"
