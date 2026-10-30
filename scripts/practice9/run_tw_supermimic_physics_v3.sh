#!/usr/bin/env bash
# Bounded method-transfer experiment. Does not promote the resulting policy.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export TW_SUPERMIMIC_MJCF="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml"
export MUJOCO_GL=egl
STATE="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/physics_v3_1000"
mkdir -p "$STATE"
exec 9>"$STATE/worker.lock"
flock -n 9 || { echo 'Pilot worker already active' >&2; exit 1; }
[[ ! -e "$STATE/status" ]] || { echo 'Existing run: inspect/resume checkpoints explicitly' >&2; exit 1; }
printf '%s\n' running > "$STATE/status"
trap 'printf "failed\n" > "$STATE/status"' ERR
git rev-parse HEAD > "$STATE/source_commit"
git diff --binary > "$STATE/source_dirty.patch"
python -m pip freeze > "$STATE/runtime_freeze.txt"
python -u -m supermimic.scripts.train_tw \
 --manifest "$GPUFREE_DATA_ROOT/datasets/practice9/training_speed_blend_20261026/train.json" \
 --num-envs 256 --seed 42 --log-dir "$GPUFREE_DATA_ROOT/twrl_runs/supermimic_tw_20261031" \
 --experiment-name physics_v3_1000 --sm.num-iterations 1000 --sm.lr-decay-iters 1000 \
 --sm.wm-batch-size 512 --sm.wm-updates-per-iter 10 --sm.policy-batch-size 512 \
 --sm.policy-chunk-size 128 --sm.log-interval 10 --sm.save-interval 100 \
 --sm.buffer-steps 1000 2>&1 | tee "$STATE/train.log"
python - "$STATE" <<'PYCHECK'
from pathlib import Path
import sys,torch
from supermimic.config.tw.velocity_limits import PHYSICS_CONTRACT
state=Path(sys.argv[1]);root=Path('/root/gpufree-data/twrl_runs/supermimic_tw_20261031/physics_v3_1000')
files=list(root.glob('*/checkpoint_final.pt'));assert len(files)==1,files
checkpoint=torch.load(files[0],map_location='cpu',weights_only=False)
assert checkpoint['iteration']==1000
assert checkpoint['cfg']['extra']['tw_physics_contract']==PHYSICS_CONTRACT
assert not any(key in checkpoint['cfg']['extra'] for key in ('tw_context_migration','tw_replay_migration','tw_history_migration'))
(state/'run_dir').write_text(str(files[0].parent)+'\n')
PYCHECK
printf '%s\n' trained_not_evaluated > "$STATE/status"
