#!/usr/bin/env bash
set -eo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/locomotion/remote_env.sh
root=/root/gpufree-data/tw_stage1/issue_runs_20261108
for seed in 1 2 3; do
 python -u scripts/locomotion/eval_cmd.py --task Unitree-TW-30dof-Velocity --headless --num_envs 128 --seed "$seed" --checkpoint validation_artifacts/tw_stage1_ft5_20261107/model_5997.pt --output "$root/ft5_seed${seed}.json" > "$root/ft5_seed${seed}_v3.log" 2>&1
done
printf 'complete\n' > "$root/baseline_complete"
