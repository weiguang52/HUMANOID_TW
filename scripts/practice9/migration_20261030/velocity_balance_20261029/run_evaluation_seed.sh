#!/bin/bash
set -eo pipefail
ROOT=/root/gpufree-data/datasets/practice9
BASE=$ROOT/velocity_balance_20261029
SEED=$1; GPU=$2
if [[ "$SEED" = 42 ]]; then D=$BASE; else D=$ROOT/velocity_balance_seed123_20261029; fi
exec > "$D/evaluation_supervisor.log" 2>&1
exec 8>"$D/evaluation.lock"; flock -n 8
trap 'echo failed > "$D/evaluation_status"' ERR
PY=/root/gpufree-data/conda_envs/env_isaaclab/bin/python
"$PY" -c 'import json,sys;assert json.load(open(sys.argv[1]))["passed"]' "$D/formal_audit.json"
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
cd /root/gpufree-data/projects/HUMANOID_TW
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=.5 P9_CONTACT_POINT_TELEMETRY=1
unset P9_LOCOMOTION_DAMPING P9_PD_DAMPING_MULT P9_PD_STIFFNESS_MULT P9_ANKLE_YAW_DAMPING_MULT
unset P9_TARGET_VELOCITY_SCALE P9_TARGET_SMOOTHING_TAU P9_TARGET_LIMIT_VELOCITY
export PYTHONPATH="$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
CKPT=$(cat "$D/actual_formal/final_checkpoint")
EXPECTED=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "$D/formal_audit.json")
test "$(basename "$CKPT")" = model_28494.pt
test "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$EXPECTED"
printf '%s\n' "$CKPT" > "$D/checkpoint"
python - "$D" "$CKPT" "$EXPECTED" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);(r/'checkpoint_freeze.json').write_text(json.dumps({'checkpoint':sys.argv[2],'sha256':sys.argv[3],'selection':'fixed final500 only','policy_promoted':False},indent=2))
PY
echo retention > "$D/evaluation_status"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=$ROOT/tw198_v1/manifest.json
python scripts/rsl_rl/play.py --headless --enable_cameras --robot_usd "$ROOT/head_locked_20261024/robot_colored.usd" --task Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D --num_envs 1 --checkpoint "$CKPT" --seed 42 --standing_diagnostic --evaluation_batch "$D/retention/jobs.json" > "$D/retention/run.log" 2>&1
test "$(cat "$D/retention/jobs.json.done")" = completed
while [[ "$(cat "$BASE/accepted_baseline/status" 2>/dev/null || true)" != completed ]]; do
 [[ "$(cat "$BASE/accepted_baseline/status" 2>/dev/null || true)" != failed ]] || exit 1
 tmux has-session -t contact_history_baseline || { echo 'Baseline process missing'; exit 1; }
 sleep 10
done
if ! python "$BASE/audit_retention.py" "$D" > "$D/retention_audit.log" 2>&1; then
 echo retention_rejected > "$D/evaluation_status"
fi
for PHASE in evaluation independent; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$D/$PHASE/manifest.json"
 for EVAL_SEED in 42 123 2026; do
  echo "${PHASE}_${EVAL_SEED}" > "$D/evaluation_status"
  python scripts/rsl_rl/play.py --headless --robot_usd "$ROOT/head_locked_20261024/robot_colored.usd" --task Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D --num_envs 1 --checkpoint "$CKPT" --seed "$EVAL_SEED" --disable_observation_noise --evaluation_upper_body_termination --evaluation_batch "$D/$PHASE/candidate/jobs$EVAL_SEED.json" > "$D/$PHASE/candidate/seed$EVAL_SEED.log" 2>&1
  test "$(cat "$D/$PHASE/candidate/jobs$EVAL_SEED.json.done")" = completed
 done
 python "$D/$PHASE/analyze.py" > "$D/$PHASE/analysis.log"
 if [[ "$PHASE" = evaluation ]]; then
  python "$D/$PHASE/summarize_flat.py" > "$D/$PHASE/flat.log"
  python "$D/$PHASE/locomotion_fidelity.py" > "$D/$PHASE/fidelity.log"
 else
  python "$D/$PHASE/summarize_outcomes.py" > "$D/$PHASE/outcomes.log"
 fi
done
echo completed > "$D/evaluation_status"
