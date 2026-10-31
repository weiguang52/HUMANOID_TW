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
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
S="$R/response_pose_fit_v1"
C="$R/standing_wm_calibration_v1/fit/world_model_calibrated.pt"
python -c 'import json; assert json.load(open("/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/response_pose_fit_v1/smoke_integrity.json"))["passed"]'
printf 'running\n' > "$S/status"
trap 'printf "failed\n" > "$S/status"' ERR
mkdir "$S/source_snapshot"
git rev-parse HEAD > "$S/source_commit"
git diff --binary > "$S/source_dirty.patch"
cp scripts/practice9/calibrate_supermimic_branch_pair.py scripts/practice9/supermimic_antithetic_response.py scripts/practice9/supermimic_pose_integration_diagnostic.py "$S/source_snapshot/"
ln -s "$R/antithetic_response_fit_v1/candidate" "$S/control"
python -u scripts/practice9/calibrate_supermimic_branch_pair.py --parent /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/standing_wm_calibration_v1/fit/world_model_calibrated.pt --collection /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_perturb_v1 --branches /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/orthogonal_branches_v1 --arm candidate --response-coefficient 1 --response-fit-audit /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/antithetic_response_fit_v1/pair_integrity.json --pose-integration-fit-audit /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/antithetic_integration_consistency_v1/audit.json --output "$S/candidate/fit" > "$S/candidate.fit.log" 2>&1
python scripts/practice9/audit_supermimic_offline_calibration_integrity.py --artifact "$S/candidate/fit/world_model_calibrated.pt" --source-checkpoint "$C" --collection "$R/static_perturb_v1" --output "$S/candidate/integrity.json" > "$S/candidate.integrity.log" 2>&1
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$S/candidate/fit/world_model_calibrated.pt" --output "$S/candidate/after.json" > "$S/candidate.audit.log" 2>&1
python scripts/practice9/compare_supermimic_calibration.py --before "$R/standing_wm_calibration_v1/after.json" --after "$S/candidate/after.json" --output "$S/candidate/initial_h8_gate.json" --nonregression-only > "$S/candidate.gate.log" 2>&1
python scripts/practice9/compare_supermimic_calibration.py --before "$S/control/after.json" --after "$S/candidate/after.json" --output "$S/paired_trajectory_gate.json" --nonregression-only > "$S/paired_gate.log" 2>&1
python scripts/practice9/audit_supermimic_branch_pair.py --pair-root "$S" --branches "$R/orthogonal_branches_v1" --asset /root/gpufree-data/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml --parent "$C" > "$S/branch_validation.log" 2>&1
printf 'audited_pending_review\n' > "$S/status"
