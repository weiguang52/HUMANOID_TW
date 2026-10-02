#!/usr/bin/env bash
set -euo pipefail
DATA=/root/gpufree-data
REPO="$DATA/projects/HUMANOID_TW"
STATE="$DATA/datasets/practice9/s2_simple_v1"
export CUDA_VISIBLE_DEVICES="${S2_GPU:-1}"
export PYTHONPATH="$REPO/source/unitree_rl_lab:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$DATA/isaacsim:${LD_LIBRARY_PATH:-}"
set +u
source /opt/conda/etc/profile.d/conda.sh
conda activate "$DATA/conda_envs/env_isaaclab"
set -u
cd "$REPO"
trap 'code=$?; if ((code)); then echo failed:$code > "$STATE/evaluation.status"; fi' EXIT
echo running > "$STATE/evaluation.status"
python - "$STATE" <<'PYMANIFEST'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.load(open(p/'manifest.json'))
for r in m['motions']:
 if not r['quality_pass'] or not r['fk_quality_pass']: raise ValueError('Unvalidated reference')
 x=dict(m);x['motions']=[r];(p/(r['id']+'.manifest.json')).write_text(json.dumps(x,indent=2)+'\n')
PYMANIFEST
CHECKPOINT=$(cat "$DATA/datasets/practice9/tw59_s1/checkpoint")
for ID in s2_stand s2_weight_shift s2_wave_000113; do
 export PRACTICE9_CUSTOM_MOTION_MANIFEST="$STATE/$ID.manifest.json"
 python scripts/rsl_rl/play.py --headless --task Unitree-Custom-Humanoid-30dof-Mimic-HumanML3D --num_envs 1 --checkpoint "$CHECKPOINT" --evaluation_motion_id 0 --evaluation_steps 1000 --seed 42 --disable_observation_noise --evaluation_output "$STATE/$ID.evaluation.json" --telemetry_output "$STATE/$ID.telemetry.npz" > "$STATE/$ID.evaluation.log" 2>&1
done
echo completed > "$STATE/evaluation.status"
