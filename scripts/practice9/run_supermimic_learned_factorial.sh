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
probe_root="$probe_data_root/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics"
export TW_SUPERMIMIC_MJCF="$probe_data_root/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml"
case "${1:-}" in
 smoke) probe_args=(--num-envs 4 --max-steps 40 --smoke); probe_name=smoke ;;
 repeat) probe_args=(--num-envs 32 --max-steps 1000 --pd-repeat); probe_name=repeat_pd ;;
 formal) probe_args=(--num-envs 32 --max-steps 1000); probe_name=probe ;;
 *) echo 'Specify smoke or formal' >&2; exit 1 ;;
esac
mkdir -p "$probe_root/learned_reset_sensor_probe_v1"
exec 9>"$probe_root/learned_reset_sensor_probe_v1/$probe_name.lock"
flock -n 9 || exit 1
trap 'code=$?; if (( code )); then echo "failed_exit_$code" > "$probe_root/learned_reset_sensor_probe_v1/${probe_name}_status"; fi' EXIT
echo running > "$probe_root/learned_reset_sensor_probe_v1/${probe_name}_status"
python -u scripts/practice9/collect_supermimic_learned_factorial.py --manifest "$probe_data_root/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json" --checkpoint "$probe_root/real_ppo_standing_v1/checkpoint_final.pt" --output "$probe_root/learned_reset_sensor_probe_v1/$probe_name" "${probe_args[@]}"

echo completed_pending_independent_audit > "$probe_root/learned_reset_sensor_probe_v1/${probe_name}_status"
