#!/usr/bin/env bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
python scripts/practice9/retarget_tw75_batches.py --worker "$1"
