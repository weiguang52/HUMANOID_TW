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
B=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031
S="$B/longterm_diagnostics/real_ppo_multiseed_standing_v1"
parent=/root/gpufree-data/twrl_runs/supermimic_tw_20261031/standing_velocity_v2_seed42/2026-10-31_09-24-01/checkpoint_final.pt
manifest="$B/static_probe/manifest.json"
exec 9>"$S/runner.lock"
flock -n 9 || exit 1
trap 'code=$?; if (( code )); then printf "failed_exit_%s_preserve_checkpoints\n" "$code" > "$S/status"; fi' EXIT
python - "$S" <<'PY'
import json,sys,hashlib
from pathlib import Path
s=Path(sys.argv[1]);d=json.loads((s/'seed_equivalence_v2.json').read_text())
assert d['default42_source_ast_equivalence_passed']
assert all(x['audit']['passed'] for x in d['artifact_audits'])
assert all(x['audit']['passed'] for x in d['reused_seed42'])
for x in d['reused_seed42']:assert hashlib.sha256((Path(x['path'])/'checkpoint_final.pt').read_bytes()).hexdigest()==x['sha256']
# Preserve this failure explicitly. This runner does not certify acceptance.
assert d['runtime_exact_regression_passed'] is False
PY
printf 'training_and_evaluating_not_promoted\n' > "$S/status"
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
for seed in 43 44; do
 for arm in baseline smooth; do
  run="$S/${arm}_seed${seed}"
  python - <<'PY'
import shutil
assert shutil.disk_usage('/root/gpufree-data').free > 10*1024**3, 'Data disk below reserved10GiB; preserve all checkpoints and inspect'
PY
  mkdir -p "$run"
  if [[ ! -f "$run/checkpoint_final.pt" ]]; then
   [[ ! -f "$run/metrics.jsonl" ]] || { echo 'Incomplete run: inspect and explicitly resume latest completed update; no restart' >&2; exit 1; }
   args=()
   if [[ "$arm" == smooth ]]; then args+=(--sole-normal-penalty 2 --sole-normal-loss smooth_distance); fi
   export TW_SUPERMIMIC_MJCF="$B/robot/tw30.xml"
   printf 'training\n' > "$run/experiment_status"
   python -u scripts/practice9/train_supermimic_real_ppo.py --parent "$parent" --manifest "$manifest" --output "$run" --num-envs 256 --updates 1000 --seed "$seed" "${args[@]}" > "$run/worker.log" 2>&1
  fi
  if [[ ! -f "$run/artifact_integrity.json" ]]; then
   python scripts/practice9/audit_supermimic_real_ppo_artifact.py --run "$run" --parent "$parent" --manifest "$manifest" --expected-updates 1000 --expected-envs 256 --expected-seed "$seed" --output "$run/artifact_integrity.json" > "$run/artifact_audit.log" 2>&1
  fi
  export TW_SUPERMIMIC_MJCF="$B/robot/tw30_colored.xml"
  if [[ ! -f "$run/evaluation/results.json" ]]; then
   [[ ! -d "$run/evaluation" ]] || { echo 'Incomplete evaluation must be inspected, never overwrite' >&2; exit 1; }
   printf 'nominal_evaluation\n' > "$run/experiment_status"
   python -u scripts/practice9/evaluate_supermimic_tw.py --checkpoint "$run/checkpoint_final.pt" --manifest "$manifest" --output "$run/evaluation" --max-steps 1000 --seed 42 --agents trained zero --video --native-contacts > "$run/evaluation.log" 2>&1
  fi
  if [[ ! -f "$run/evaluation/episode_media_integrity.json" ]]; then
   python scripts/practice9/verify_supermimic_episode_media.py --evaluation "$run/evaluation" > "$run/media_audit.log" 2>&1
  fi
  export TW_SUPERMIMIC_MJCF="$B/robot/tw30.xml"
  if [[ ! -f "$run/factorial/results.json" ]]; then
   [[ ! -d "$run/factorial" ]] || { echo 'Incomplete factorial must be inspected, never overwrite' >&2; exit 1; }
   printf 'factorial_evaluation\n' > "$run/experiment_status"
   python -u scripts/practice9/collect_supermimic_learned_factorial.py --manifest "$manifest" --checkpoint "$run/checkpoint_final.pt" --output "$run/factorial" --num-envs 32 --max-steps 1000 > "$run/factorial.log" 2>&1
  fi
  if [[ ! -f "$run/factorial/independent_integrity.json" ]]; then
   if python scripts/practice9/audit_supermimic_learned_factorial.py --root "$run/factorial" --checkpoint "$run/checkpoint_final.pt" > "$run/factorial_audit.log" 2>&1; then
    printf 'evaluated_pending_visual_review\n' > "$run/experiment_status"
   else
    # Pairing can fail just as in the preserved seed42 experiment. Never relabel it.
    python - "$run/factorial/independent_integrity.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['archive_invariants_passed'] and not p['numeric_pairing_passed']
PY
    printf 'evaluated_numeric_pairing_failed_pending_review\n' > "$run/experiment_status"
   fi
  fi
 done
done
printf 'all_runs_evaluated_pending_multiseed_visual_and_physical_review\n' > "$S/status"
