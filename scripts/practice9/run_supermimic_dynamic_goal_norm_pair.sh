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

R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
O="$R/dynamic_goal_norm_pair_v1"
exec 9>"$O/runner.lock"
flock -n 9 || exit 1
trap 'code=$?; if (( code )); then printf "failed_%s_preserve_checkpoints\n" "$code" > "$O/status"; fi' EXIT
parent=/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt
manifest=/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/train.json
test_manifest=/root/gpufree-data/datasets/practice9/independent_holdout_20261025/test.json
git rev-parse HEAD > "$O/source_commit"
git diff --binary > "$O/source_dirty.patch"
for arm in baseline calibrated; do
 python -c 'import shutil; assert shutil.disk_usage("/root/gpufree-data").free>10*1024**3'
 args=()
 if [[ "$arm" == calibrated ]]; then args+=(--goal-normalizer-calibration "$R/training_goal_norm_calibration_v1/goal_normalizer.pt"); fi
 export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml
 printf 'training_%s\n' "$arm" > "$O/status"
 python -u scripts/practice9/train_supermimic_real_ppo.py --parent "$parent" --manifest "$manifest" --output "$O/$arm" --num-envs 256 --updates 1000 --seed 42 "${args[@]}" > "$O/${arm}.log" 2>&1
 python scripts/practice9/audit_supermimic_real_ppo_artifact.py --run "$O/$arm" --parent "$parent" --manifest "$manifest" --expected-updates 1000 --expected-envs 256 --expected-seed 42 --output "$O/${arm}_integrity.json" > "$O/${arm}_audit.log" 2>&1
 printf 'evaluating_%s\n' "$arm" > "$O/status"
 export TW_SUPERMIMIC_MJCF=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml
 python -u scripts/practice9/evaluate_supermimic_tw.py --checkpoint "$O/$arm/checkpoint_final.pt" --manifest "$test_manifest" --output "$O/$arm/evaluation" --max-steps 1000 --seed 42 --agents trained --video --native-contacts > "$O/${arm}_evaluation.log" 2>&1
 python scripts/practice9/verify_supermimic_episode_media.py --evaluation "$O/$arm/evaluation" > "$O/${arm}_media_audit.log" 2>&1
done
printf 'pilot_complete_pending_all_motion_and_visual_review_not_acceptance\n' > "$O/status"
