#!/usr/bin/env bash
# Bounded method-transfer experiment. Does not promote the resulting policy.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
export TW_SUPERMIMIC_MJCF="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/robot/tw30.xml"
export MUJOCO_GL=egl
STATE="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/pilot_replayfix_1700"
mkdir -p "$STATE"
exec 9>"$STATE/worker.lock"
flock -n 9 || { echo 'Pilot worker already active' >&2; exit 1; }
[[ ! -e "$STATE/status" ]] || { echo 'Existing run: inspect/resume checkpoints explicitly' >&2; exit 1; }
printf '%s\n' running > "$STATE/status"
trap 'printf "failed\n" > "$STATE/status"' ERR
git rev-parse HEAD > "$STATE/source_commit"
git diff --binary > "$STATE/source_dirty.patch"
python -m pip freeze > "$STATE/runtime_freeze.txt"
read -r WM_RATE ACTOR_RATE CRITIC_RATE < <(python - "$STATE" <<'PYRATE'
import torch,json,sys
from pathlib import Path
p=Path('/root/gpufree-data/twrl_runs/supermimic_tw_20261031/pilot_contextfix_1000/2026-10-31_03-23-24/checkpoint_0000700.pt')
c=torch.load(p,map_location='cpu',weights_only=False);assert c['iteration']==700
rates=[float(c[k]['param_groups'][0]['lr']) for k in ('wm_optimizer','actor_optimizer','critic_optimizer')]
assert all(rate>0 for rate in rates)
(Path(sys.argv[1])/'continuation_schedule.json').write_text(json.dumps(dict(source_iteration=700,target_iteration=1700,fully_corrected_updates=1000,schedule='constant at restored optimizer LR; no upward LR reset',rates=rates),indent=2)+'\n')
print(*rates)
PYRATE
)
python -u -m supermimic.scripts.train_tw \
 --manifest "$GPUFREE_DATA_ROOT/datasets/practice9/training_speed_blend_20261026/train.json" \
 --num-envs 256 --seed 42 --log-dir "$GPUFREE_DATA_ROOT/twrl_runs/supermimic_tw_20261031" \
 --experiment-name pilot_replayfix_1700 --resume "$GPUFREE_DATA_ROOT/twrl_runs/supermimic_tw_20261031/pilot_contextfix_1000/2026-10-31_03-23-24/checkpoint_0000700.pt" --resume-replay-migration --sm.num-iterations 1700 --sm.lr-decay-iters 0 --sm.wm-lr "$WM_RATE" --sm.actor-lr "$ACTOR_RATE" --sm.critic-lr "$CRITIC_RATE" \
 --sm.wm-batch-size 512 --sm.wm-updates-per-iter 10 --sm.policy-batch-size 512 \
 --sm.policy-chunk-size 128 --sm.log-interval 10 --sm.save-interval 100 \
 --sm.buffer-steps 1000 2>&1 | tee "$STATE/train.log"
printf '%s\n' trained_not_evaluated > "$STATE/status"
