#!/usr/bin/env bash
# Fixed seeds/budgets; collect all outcomes, never promote a policy.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
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

export XDG_DATA_HOME=/root/gpufree-data/share
export MPLCONFIGDIR=/root/gpufree-data/cache/matplotlib
export WANDB_DIR=/root/gpufree-data/logs/wandb

export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
python -u scripts/practice9/run_supermimic_cold_authority_repeats.py > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/balance_authority_v1/cold_repeats_supervisor.log 2>&1
