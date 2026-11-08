#!/bin/bash
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export PATH=/root/gpufree-data/conda_envs/env_isaaclab/bin:$PATH
export CONDA_PREFIX=/root/gpufree-data/conda_envs/env_isaaclab
source /root/gpufree-data/isaacsim/setup_conda_env.sh
for d in /root/gpufree-data/conda_envs/env_isaaclab/lib/python3.11/site-packages/nvidia/*/lib /root/gpufree-data/conda_envs/env_supermimic/lib/python3.11/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"
done
cd /root/gpufree-data/projects/HUMANOID_TW
export TW_RESET_STD=0.3
export PYTHONPATH=/root/gpufree-data/projects/HUMANOID_TW/source/unitree_rl_lab:$PYTHONPATH
python scripts/locomotion/train.py --task Unitree-TW-30dof-Velocity --headless --num_envs 4096 --max_iterations 1500 --run_name tw_stage1_s1_ft5 --seed 1 --resume --load_run 2026-11-07_16-32-16_tw_stage1_s1_ft4 --checkpoint model_4498.pt > /root/gpufree-data/tw_stage1/train1500_s1_ft5.log 2>&1
echo "EXIT $?" >> /root/gpufree-data/tw_stage1/train1500_s1_ft5.log