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
probe_root="$probe_data_root/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/sole_normal_ppo_v1"
manifest="$probe_data_root/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json"
export TW_SUPERMIMIC_MJCF="$probe_data_root/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml"
exec 9>"$probe_root/factorial.lock"
flock -n 9 || exit 1
trap 'code=$?; if (( code )); then echo "failed_exit_$code" > "$probe_root/factorial_status"; fi' EXIT
echo running > "$probe_root/factorial_status"
python -u scripts/practice9/collect_supermimic_learned_factorial.py --manifest "$manifest" --checkpoint "$probe_root/trained/checkpoint_final.pt" --output "$probe_root/factorial" --num-envs 32 --max-steps 1000
echo completed_pending_archive_and_pair_review > "$probe_root/factorial_status"
