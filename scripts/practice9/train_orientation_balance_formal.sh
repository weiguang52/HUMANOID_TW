#!/bin/bash
set -euo pipefail
GPU=${1:?GPU index}; VARIANT=${2:?moving}; MODE=${3:-formal}
[[ "$MODE" == formal ]] || exit 2
[[ "$VARIANT" == actual ]] || exit 2
KIND=MovingSlip
case "$MODE" in probe) ENVS=64; ITERATIONS=20;; smoke) ENVS=64; ITERATIONS=2;; reload) ENVS=64; ITERATIONS=1;; formal) ENVS=1024; ITERATIONS=500;; *) exit 2;; esac
REPO=/root/gpufree-data/projects/HUMANOID_TW
STATE=/root/gpufree-data/datasets/practice9/orientation_balance_20261029
OUT="$STATE/${VARIANT}_${MODE}"
mkdir -p "$OUT"
exec 8>"$OUT/run.lock"; flock -n 8
[[ ! -e "$OUT/status" ]] || { echo 'Refusing to overwrite an existing run'; exit 1; }
exec > >(tee -a "$OUT/supervisor.log") 2>&1
trap 'echo failed > "$OUT/status"' ERR
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
set -u
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/training_speed_blend_20261026/train.json
export P9_TARGET_SMOOTHING_TAU=0 P9_TARGET_LIMIT_VELOCITY=1 P9_TARGET_VELOCITY_SCALE=0.5
export P9_PD_STIFFNESS_MULT=1 P9_PD_DAMPING_MULT=1 P9_CONTACT_PHASE_WEIGHT=0.5
MEMORY=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( MEMORY < 256 )) || { echo 'GPU occupied'; exit 3; }
if [[ "$MODE" == formal ]]; then
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["all_passed"]' "$STATE/unit_suite_audit.json"
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$STATE/smoke_likelihood_audit.json"
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$STATE/smoke_deployment_audit.json"
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$STATE/reload_audit.json"
 test "$(cat "$STATE/${VARIANT}_smoke/status")" = trained
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$STATE/smoke_audit.json"
 python -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$STATE/smoke_runtime_audit.json"
fi
INITIAL=$(cat /root/gpufree-data/datasets/practice9/head_locked_20261024/final_checkpoint)
TEACHER="$INITIAL"
python -c 'import sys,hashlib;assert hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()=="9f06522c8b5514d7d334f662b9aeb70eb541a190d8d13a54c507998ce2aae872"' "$TEACHER"
if [[ "$MODE" == reload ]]; then INITIAL=$(cat "$STATE/actual_smoke/final_checkpoint"); fi
test -s "$INITIAL"
FRAMES=$(python -c 'import json,os;print(sum(r["frames"] for r in json.load(open(os.environ["PRACTICE9_CUSTOM_MOTION_MANIFEST"]))["motions"]))')
EXPERIMENT=/root/twrl_runs/orientation_balance_20261029/logs
RUN_NAME=orientation_balance_20261029_${VARIANT}_${MODE}
mkdir -p "$EXPERIMENT"
if [[ ! -e "$EXPERIMENT/$(basename "$(dirname "$INITIAL")")" ]]; then ln -s "$(dirname "$INITIAL")" "$EXPERIMENT/$(basename "$(dirname "$INITIAL")")"; fi
git rev-parse HEAD > "$OUT/source_commit"
sha256sum "$INITIAL" "$PRACTICE9_CUSTOM_MOTION_MANIFEST" > "$OUT/input_sha256.txt"
printf '%s\n' "$INITIAL" > "$OUT/initial_checkpoint"
TEACHER_EXPORT=/root/gpufree-data/datasets/practice9/frozen_teachers/head_locked_model27995_20261028/policy.pt
FORK_ARGS=(--allow_physics_slip_load_change)
if [[ "$MODE" == reload ]]; then FORK_ARGS=(); fi
RESUME_ARGS=(${FORK_ARGS[@]} --physics_slip_minimum_load 5.0 --physics_slip_history_samples 3 --train_controlled_action_likelihood --train_candidate_action_scope lower_body_waist --train_reference_speed_teacher_export "$TEACHER_EXPORT" --physics_mean_slip --train_ankle_filter_alpha 0.5 --restore_optimizer_learning_rate --policy_anchor_coefficient 2.0 --contact_feedback_mode "$VARIANT" --policy_anchor_temporal_coefficient 1.0 --policy_anchor_temporal_order 2 --policy_anchor_checkpoint "$TEACHER" --policy_anchor_temporal_target student_mean)

sha256sum source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/lookback_sampling.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/commands.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/adaptive_sampling.py scripts/rsl_rl/lookback_contract.py scripts/practice9/test_lookback_sampling.py scripts/rsl_rl/contact_history.py scripts/rsl_rl/contact_feedback.py scripts/practice9/test_contact_history.py scripts/rsl_rl/physics_slip_reward.py scripts/practice9/test_load_aligned_slip.py scripts/rsl_rl/controlled_action_likelihood.py scripts/rsl_rl/controlled_likelihood_ppo.py scripts/practice9/test_controlled_action_likelihood.py scripts/practice9/test_controlled_activity.py scripts/rsl_rl/candidate_action_scope.py scripts/rsl_rl/candidate_scope_contract.py scripts/rsl_rl/batch_mimic_evaluation.py scripts/rsl_rl/play.py scripts/practice9/test_scoped_training_speed_blend.py scripts/rsl_rl/training_speed_blend.py scripts/rsl_rl/reference_speed_blend.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/leg_position_error.py scripts/rsl_rl/second_order_anchor.py scripts/rsl_rl/temporal_target_contract.py scripts/rsl_rl/contact_feedback.py scripts/rsl_rl/contact_policy.py scripts/rsl_rl/anchored_ppo.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/leg_velocity_error.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/point_slip.py source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/custom_30dof/humanml3d/point_slip_env_cfg.py scripts/rsl_rl/train.py "$0" > "$OUT/code_sha256.txt"
python -c 'import shutil,sys;assert shutil.disk_usage(sys.argv[1]).free>1500000000' "$STATE"
echo training > "$OUT/status"
python scripts/rsl_rl/train.py "${RESUME_ARGS[@]}" --headless --resume --load_run "$(basename "$(dirname "$INITIAL")")" --checkpoint "$(basename "$INITIAL")" --task "Unitree-Custom-Humanoid-30dof-${KIND}-HumanML3D" --seed 42 --logger tensorboard --num_envs "$ENVS" --max_iterations "$ITERATIONS" --run_name "$RUN_NAME" "env.commands.motion.max_motion_frames=$FRAMES" env.commands.motion.adaptive_uniform_ratio=0.8 env.commands.motion.adaptive_lookback_share=0.5 env.rewards.horizontal_path_error.weight=-1.0 env.rewards.motion_global_anchor_ori.weight=1.125 agent.algorithm.lam=0.99 "agent.experiment_name=$EXPERIMENT" agent.save_interval=100 env.rewards.moving_point_slip.weight=-1.0 env.rewards.action_rate_l2.weight=-0.3 env.rewards.moving_leg_velocity.weight=-0.6 env.rewards.moving_leg_position.weight=-0.35 > "$OUT/train.log" 2>&1
RUN=$(find "$EXPERIMENT" -maxdepth 1 -type d -name "*_$RUN_NAME" | sort | tail -1)
CHECKPOINT=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -s "$CHECKPOINT"; realpath "$CHECKPOINT" > "$OUT/final_checkpoint"
echo trained > "$OUT/status"
