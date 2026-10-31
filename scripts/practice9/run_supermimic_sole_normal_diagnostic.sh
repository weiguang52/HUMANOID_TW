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
parent="$probe_data_root/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt"
manifest="$probe_data_root/datasets/practice9/supermimic_tw_20261031/static_probe/manifest.json"
export TW_SUPERMIMIC_MJCF="$probe_data_root/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml"
case "${1:-}" in
 smoke) run_name=smoke; run_args=(--num-envs 8 --updates 4 --stop-after 2) ;;
 resume_smoke) run_name=smoke; run_args=(--num-envs 8 --updates 4 --resume "$probe_root/smoke/checkpoint_0002.pt") ;;
 formal) run_name=trained; run_args=(--num-envs 256 --updates 1000) ;;
 *) echo 'Specify smoke, resume_smoke or formal' >&2; exit 1 ;;
esac
state="$probe_root/$run_name"
mkdir -p "$state"
exec 9>"$state/runner.lock"
flock -n 9 || exit 1
trap 'code=$?; if (( code )); then echo "failed_exit_$code" > "$state/experiment_status"; fi' EXIT
if [[ ! -f "$state/checkpoint_final.pt" ]]; then
 if [[ "$run_name" == trained && -f "$state/metrics.jsonl" ]]; then
  latest=$(find "$state" -maxdepth 1 -name 'checkpoint_[0-9][0-9][0-9][0-9].pt' | sort | tail -n1)
  [[ -n "$latest" ]] || { echo 'Inspect incomplete run without a completed checkpoint' >&2; exit 1; }
  run_args+=(--resume "$latest")
 fi
 echo training > "$state/experiment_status"
 python -u scripts/practice9/train_supermimic_real_ppo.py --parent "$parent" --manifest "$manifest" --output "$state" --sole-normal-penalty 2 "${run_args[@]}"
fi
if [[ "$run_name" == smoke ]]; then
 echo smoke_segment_complete > "$state/experiment_status"
 exit 0
fi
if [[ ! -f "$state/artifact_integrity.json" ]]; then
python scripts/practice9/audit_supermimic_real_ppo_artifact.py --run "$state" --parent "$parent" --manifest "$manifest" --expected-updates 1000 --expected-envs 256 --output "$state/artifact_integrity.json"
fi
export TW_SUPERMIMIC_MJCF="$probe_data_root/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml"
if [[ ! -f "$state/evaluation/results.json" ]]; then
 [[ ! -d "$state/evaluation" ]] || { echo 'Inspect preserved incomplete evaluation' >&2; exit 1; }
 echo evaluating > "$state/experiment_status"
 python -u scripts/practice9/evaluate_supermimic_tw.py --checkpoint "$state/checkpoint_final.pt" --manifest "$manifest" --output "$state/evaluation" --max-steps 1000 --seed 42 --agents trained zero --video --native-contacts
fi
if [[ ! -f "$state/evaluation/episode_media_integrity.json" ]]; then
python scripts/practice9/verify_supermimic_episode_media.py --evaluation "$state/evaluation"
fi
echo evaluated_pending_visual_and_contact_review > "$state/experiment_status"
