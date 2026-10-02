#!/usr/bin/env bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
ROOT=/root/gpufree-data/datasets/practice9/tw75_dataset_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/practice9/retarget_humanml3d.py --input "$ROOT/pilot.inputs.txt" --output-dir "$ROOT/pilot_$1/retargeted" --retarget-profile "$ROOT/$1.json" --allow-quality-failures --continue-on-error
