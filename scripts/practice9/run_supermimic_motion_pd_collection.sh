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
S=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1
python -c 'import json; assert json.load(open("/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/motion_pd_collection_v1/smoke_audit_v2.json"))["passed"]'
test ! -e "$S/full"
mkdir "$S/source_snapshot"
cp scripts/practice9/collect_supermimic_motion_pd.py scripts/practice9/audit_supermimic_motion_pd.py "$S/source_snapshot/"
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
printf 'collecting\n' > "$S/status"
trap 'printf "pipeline_failed\n" > "$S/status"' ERR
python -u scripts/practice9/collect_supermimic_motion_pd.py --plan "$S/plan.json" --output "$S/full" > "$S/full.log" 2>&1
python scripts/practice9/audit_supermimic_motion_pd.py --plan "$S/plan.json" --collection "$S/full" --output "$S/full_audit.json" > "$S/full_audit.log" 2>&1
printf 'collected_audited_pending_review\n' > "$S/status"
