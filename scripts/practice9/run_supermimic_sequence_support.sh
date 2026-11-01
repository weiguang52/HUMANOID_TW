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

S=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/sequence_support_com_v1
exec 9>"$S/full_runner.lock"
flock -n 9 || exit 1
python - <<'PY'
import pathlib,json
p=pathlib.Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/sequence_support_com_v1/strict_norm_smoke/results.json');r=json.loads(p.read_text())['rows'];assert len(r)==1;v=r[0];assert v['status']=='iteration_budget_exhausted' and v['iterations']==2 and v['foot_position_vector_max_m']<=1e-5 and v['foot_orientation_vector_max_rad']<=1e-4 and v['speed_ratio']<=1+1e-5/3
PY
mkdir "$S/full_source_snapshot"
cp scripts/practice9/solve_supermimic_sequence_support.py "$S/full_source_snapshot/"
git rev-parse HEAD > "$S/full_source_snapshot/commit"
git diff --binary > "$S/full_source_snapshot/dirty.patch"
printf 'running_all_eight_diagnostic_not_training\n' > "$S/status"
trap 'printf "failed_preserve_outputs\n" > "$S/status"' ERR
python -u scripts/practice9/solve_supermimic_sequence_support.py --plan "$S/strict_norm_plan.json" --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --output "$S/full" > "$S/full.log" 2>&1
printf 'completed_pending_all_constraints_and_dynamics_review_not_acceptance\n' > "$S/status"
