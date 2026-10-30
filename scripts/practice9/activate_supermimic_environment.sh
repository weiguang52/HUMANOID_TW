#!/usr/bin/env bash
# Source this file before remote runs; keep runtime writes on the data volume.
export GPUFREE_DATA_ROOT=${GPUFREE_DATA_ROOT:-/root/gpufree-data}
[[ -x "$GPUFREE_DATA_ROOT/conda_envs/env_supermimic/bin/python" ]] || { echo "Training environment is not fully migrated" >&2; return 1; }
export HOME="$GPUFREE_DATA_ROOT/home"
export TMPDIR="$GPUFREE_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$GPUFREE_DATA_ROOT/cache"
export XDG_CONFIG_HOME="$GPUFREE_DATA_ROOT/config"
export XDG_DATA_HOME="$GPUFREE_DATA_ROOT/share"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$GPUFREE_DATA_ROOT/conda_pkgs"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export __GL_SHADER_DISK_CACHE_PATH="$XDG_CACHE_HOME/nvidia_shader"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export WANDB_DIR="$GPUFREE_DATA_ROOT/logs/wandb"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$CONDA_PKGS_DIRS" "$WANDB_DIR"
export CONDA_NO_PLUGINS=true
source /opt/conda/etc/profile.d/conda.sh
conda activate "$GPUFREE_DATA_ROOT/conda_envs/env_supermimic" || return 1
unset PYTHONHOME
export PYTHONPATH="$GPUFREE_DATA_ROOT/projects/HUMANOID_TW/third_party/supermimic_mjlab/src"
unset LD_LIBRARY_PATH
cd "$GPUFREE_DATA_ROOT/projects/HUMANOID_TW"
