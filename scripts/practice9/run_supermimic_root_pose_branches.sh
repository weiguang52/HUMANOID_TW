#!/usr/bin/env bash
set -euo pipefail
probe_data_root=/root/gpufree-data
cd "$probe_data_root/projects/HUMANOID_TW"
export PATH="$probe_data_root/conda_envs/env_supermimic/bin:$PATH"
export TMPDIR="$probe_data_root/tmp"
export XDG_CACHE_HOME="$probe_data_root/cache"
export XDG_CONFIG_HOME="$probe_data_root/config"
export PYTHONPYCACHEPREFIX="$probe_data_root/cache/pycache"
export CUDA_CACHE_PATH="$probe_data_root/cache/cuda"
export TORCH_EXTENSIONS_DIR="$probe_data_root/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$probe_data_root/cache/torchinductor"
export TORCH_HOME="$probe_data_root/cache/torch"
export __GL_SHADER_DISK_CACHE_PATH="$probe_data_root/cache/nvidia_shader"
export PYTHONPATH="$probe_data_root/projects/HUMANOID_TW/third_party/supermimic_mjlab/src"
export LD_LIBRARY_PATH="$probe_data_root/conda_envs/env_supermimic/lib/python3.11/site-packages/torch/lib:$probe_data_root/conda_envs/env_supermimic/lib"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
python -c 'import json; assert json.load(open("/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/root_pose_branches_smoke_v1/integrity.json"))["passed"]'
mkdir "$R/root_pose_branches_v1_source"
git rev-parse HEAD > "$R/root_pose_branches_v1_source/source_commit"
cp scripts/practice9/collect_supermimic_root_pose_branches.py scripts/practice9/audit_supermimic_root_pose_branches.py "$R/root_pose_branches_v1_source/"
export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
python -u scripts/practice9/collect_supermimic_root_pose_branches.py --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json --source-checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/root_pose_branches_v1
python scripts/practice9/audit_supermimic_root_pose_branches.py --root "$R/root_pose_branches_v1"
