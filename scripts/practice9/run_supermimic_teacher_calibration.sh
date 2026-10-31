#!/usr/bin/env bash
# Fixed offline experiment. Does not modify any deployable policy.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
R=/root/gpufree-data/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics
OUT=$R/wm_combined_tf_v3
exec 9>"$R/wm_combined_tf_v3.lock"
flock -n 9
trap 'rc=$?; printf "%s\n" "$rc" > "$R/wm_combined_tf_v3.exit"' EXIT
python - "$R" <<'PY'
import json,sys,datetime
from pathlib import Path
r=Path(sys.argv[1]); p=r/'wm_combined_tf_v3_protocol.json'
assert not p.exists() and not (r/'wm_combined_tf_v3').exists()
p.write_text(json.dumps(dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),change='0.5 rollout H8 loss + 0.5 real-state one-step loss; other configuration identical to combined v2',acceptance='Each motion/static group: original baseline 18 gates; v1 and v2 18 nonregression checks each, max 5%. All required.',selection='Final1000 only, no best checkpoint, fit data only, no final holdout'),indent=2))
for name in ['physics_collection_mapping_v2','static_perturb_v1']:
 import numpy as np
 c=r/name
 for row in json.loads((c/'results.json').read_text())['rows']:
  with np.load(c/row['trace']) as d:
   assert np.array_equal(d['next_state'][:-1],d['state'][1:]),row
PY
python scripts/practice9/calibrate_supermimic_world_model.py --collection "$R/combined_calibration_v2" --checkpoint /root/gpufree-data/twrl_runs/supermimic_tw_20261031/physics_v3_1000/2026-10-31_05-15-56/checkpoint_final.pt --output "$OUT" --balance-source-groups --teacher-forcing-weight 0.5
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/physics_collection_mapping_v2" --checkpoint "$OUT/world_model_calibrated.pt" --output "$OUT/motion_evaluation.json"
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$R/static_perturb_v1" --checkpoint "$OUT/world_model_calibrated.pt" --output "$OUT/static_evaluation.json"
python - "$R" <<'PY'
import json,sys,subprocess
from pathlib import Path
r=Path(sys.argv[1]);o=r/'wm_combined_tf_v3';results=[]
for group,baselines in [('motion',['calibration_model_baseline.json','wm_calibration_v1/evaluation.json','wm_combined_v2/motion_evaluation.json']),('static',['static_original_model.json','static_calibrated_model.json','wm_combined_v2/static_evaluation.json'])]:
 for i,b in enumerate(baselines):
  out=o/f'{group}_vs_{["original","v1","v2"][i]}.json'
  cmd=[sys.executable,'scripts/practice9/compare_supermimic_calibration.py','--before',str(r/b),'--after',str(o/f'{group}_evaluation.json'),'--output',str(out)]
  if i:cmd.append('--nonregression-only')
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
  result=json.loads(out.read_text());results.append(dict(comparison=out.name,passed=result['passed'],failures=[x for x in result['checks'] if not x['passed']]))
summary=dict(passed=all(x['passed'] for x in results),comparisons=results,scope='Offline development model only, no policy success')
(o/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
PY
