#!/usr/bin/env bash
# Source on the remote training host before running Isaac Lab scripts.
unset PYTHONPATH TW_RESET_STD
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export CONDA_PREFIX=/root/gpufree-data/conda_envs/env_isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
source /root/gpufree-data/isaacsim/setup_conda_env.sh
for d in "$CONDA_PREFIX"/lib/python3.11/site-packages/nvidia/*/lib /root/gpufree-data/conda_envs/env_supermimic/lib/python3.11/site-packages/nvidia/*/lib; do
  [ ! -d "$d" ] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$d"
done
export PYTHONPATH="/root/gpufree-data/projects/HUMANOID_TW/source/unitree_rl_lab:${PYTHONPATH:-}"
