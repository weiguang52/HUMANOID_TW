#!/bin/bash
set -euo pipefail
GPU=${1:?}; MODEL=${2:?}
ROOT=/root/gpufree-data/datasets/practice9/repeatability_20261025
OUT=$ROOT/$MODEL
cd /root/gpufree-data/projects/HUMANOID_TW
exec 7>"$OUT/worker.lock"; flock -n 7
[[ ! -e "$OUT/status" ]] || exit 2
trap 'echo failed > "$OUT/status"' ERR
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/gpufree-data/conda_envs/env_isaaclab
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 P9_EXPRESSION_TELEMETRY=1 P9_CONTACT_PHASE_WEIGHT=0.5
export PYTHONPATH="$PWD/source/unitree_rl_lab:$PWD/scripts/practice9:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/root/gpufree-data/isaacsim:${LD_LIBRARY_PATH:-}"
export PRACTICE9_CUSTOM_MOTION_MANIFEST=/root/gpufree-data/datasets/practice9/tw198_v1/manifest.json
CHECKPOINT=$(cat "$OUT/checkpoint")
MEMORY=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
(( MEMORY < 256 )) || exit 3
git rev-parse HEAD > "$OUT/source_commit"
echo running > "$OUT/status"
python - "$GPU" "$MODEL" "$OUT" "$CHECKPOINT" <<'PY'
import json,subprocess,sys
from pathlib import Path
_,gpu,model,out,checkpoint=sys.argv
p=Path(out)
for r in json.loads((p/'plan.json').read_text()):
 (p/'current_job').write_text(r['name'])
 camera=['--viewer_eye','1.2','0','0.15','--viewer_lookat','0','0','-0.03'] if r['front'] else []
 cmd=['python','scripts/rsl_rl/play.py','--headless','--robot_usd','/root/gpufree-data/datasets/practice9/head_locked_20261024/robot_colored.usd','--task','Unitree-Custom-Humanoid-30dof-HeadLocked-HumanML3D','--num_envs','1','--checkpoint',checkpoint,'--seed',str(r['seed']),'--standing_diagnostic','--enable_cameras','--evaluation_batch',r['jobs']]+camera
 with (p/r['name']/'run.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
PY
echo completed > "$OUT/status"
python scripts/practice9/report_repeatability.py
