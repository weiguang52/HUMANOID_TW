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
S=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/analytic_contact_balance_pair_v1
trap 'printf "failed_preserve_outputs\n" > "$S/status"' ERR
for arm in bounded_reference contact_balance; do
 python scripts/practice9/audit_supermimic_analytic_balance.py --plan "$S/plan.json" --run "$S/${arm}_smoke" --asset "$TW_SUPERMIMIC_MJCF" --output "$S/${arm}_smoke_audit.json" > "$S/${arm}_smoke_audit.log" 2>&1
done
mkdir "$S/source_snapshot"
cp scripts/practice9/collect_supermimic_analytic_balance.py scripts/practice9/audit_supermimic_analytic_balance.py scripts/practice9/supermimic_contact_balance.py "$S/source_snapshot/"
git rev-parse HEAD > "$S/source_snapshot/commit"
git diff --binary > "$S/source_snapshot/dirty.patch"
for arm in bounded_reference contact_balance; do
 printf 'collecting_%s\n' "$arm" > "$S/status"
 python -u scripts/practice9/collect_supermimic_analytic_balance.py --plan "$S/plan.json" --output "$S/$arm" --arm "$arm" > "$S/${arm}.log" 2>&1
 python scripts/practice9/audit_supermimic_analytic_balance.py --plan "$S/plan.json" --run "$S/$arm" --asset "$TW_SUPERMIMIC_MJCF" --output "$S/${arm}_audit.json" > "$S/${arm}_audit.log" 2>&1
done
printf 'pair_collected_audited_pending_physical_review_not_acceptance\n' > "$S/status"
