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

cd /root/gpufree-data/projects/HUMANOID_TW



export CUBLAS_WORKSPACE_CONFIG=:4096:8
wide_pair_state=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1
trap 'printf "failed\n" > "$wide_pair_state/status"' ERR
python - <<'CHECK_SMOKE'
import json
from pathlib import Path
r=Path('/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1')
assert json.loads((r/'smoke4/fit_integrity.json').read_text())['passed']
assert not (r/'control/fit').exists() and not (r/'candidate/fit').exists()
CHECK_SMOKE
git rev-parse HEAD > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/source_commit
echo running > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/status
python -u scripts/practice9/calibrate_supermimic_static_coupled_pair.py --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/fit --enabled 0 --updates 1000 > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control_fit.log 2>&1
python -u scripts/practice9/calibrate_supermimic_static_coupled_pair.py --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/fit --enabled 1 --updates 1000 > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate_fit.log 2>&1
python scripts/practice9/audit_supermimic_static_coupled_pair.py --root /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1 --updates 1000 --reference-control /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/smoke4/repeated_control/fit/world_model_calibrated.pt
python scripts/practice9/export_supermimic_precision_artifact.py --source /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/control/fit/world_model_calibrated.pt
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_perturb_v1 --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/control/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/static_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control_static_audit.log 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/grounded_dynamic_cop_clarabel_init_v1/dense_native_collection_v1 --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/control/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/dynamic_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control_dynamic_audit.log 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/expanded_contact_sources_v1/native_full --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/control/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/expanded_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control_expanded_audit.log 2>&1
python scripts/practice9/audit_supermimic_yaw_artifact.py --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/control/fit/world_model_calibrated.pt --queries /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/yaw_canonical_contract_probe_v1/results/predictions.npz --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/control/declared_coordinate_audit.json
python scripts/practice9/export_supermimic_precision_artifact.py --source /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/candidate/fit/world_model_calibrated.pt
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_perturb_v1 --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/candidate/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/static_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate_static_audit.log 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/grounded_dynamic_cop_clarabel_init_v1/dense_native_collection_v1 --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/candidate/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/dynamic_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate_dynamic_audit.log 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/expanded_contact_sources_v1/native_full --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/candidate/fit/world_model_calibrated.pt --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/expanded_after.json > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate_expanded_audit.log 2>&1
python scripts/practice9/audit_supermimic_yaw_artifact.py --checkpoint /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/candidate/fit/world_model_calibrated.pt --queries /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/yaw_canonical_contract_probe_v1/results/predictions.npz --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/candidate/declared_coordinate_audit.json
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared --branches /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/orthogonal_branches_v1 --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_wm_calibration_v1/fit/world_model_calibrated.pt
mkdir -p /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/reference/fit
ln -s /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/expanded_native_replay_pair_v1/declared/control/fit/world_model_calibrated.pt /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/reference/fit/world_model_calibrated.pt
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared --branches /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/orthogonal_branches_v1 --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_wm_calibration_v1/fit/world_model_calibrated.pt --control-arm reference --output /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/declared/reference_branch_validation.json
echo audited_pending_review > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/status
python scripts/practice9/review_supermimic_static_coupled_metrics.py --root /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1 --static-coupled --h32 --reference-root /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/expanded_native_replay_pair_v1
printf "reviewed_no_policy_promotion\n" > /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_coupled_response_pair_v1/status
