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

export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
python -c 'import json; from pathlib import Path; r=Path("/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1"); assert all(json.load(open(r/("smoke_"+x)/"results.json"))["all_next_states_within_native_atol"] for x in ["warm","cold"])'
git rev-parse HEAD > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/source_commit
git diff --binary > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/source_dirty.patch
cp scripts/practice9/replay_supermimic_motion_pd.py /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/full_source_snapshot.py
python -u scripts/practice9/replay_supermimic_motion_pd.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1/full --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1/plan.json --phase warm --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/warm > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/warm.log 2>&1
python -u scripts/practice9/replay_supermimic_motion_pd.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1/full --manifest /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1/plan.json --phase cold --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/cold > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_replay_v1/cold.log 2>&1
