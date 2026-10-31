#!/usr/bin/env bash
# Audit both fixed models after a verified physical collection; never restart collection.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
STATE="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/longterm_diagnostics"
exec 8>"$STATE/static_model_audit.lock"
flock -n 8
python - "${1:?collector PID}" "$GPUFREE_DATA_ROOT" <<'PY'
import sys,time,json
from pathlib import Path
pid=Path('/proc')/sys.argv[1]/'cmdline';root=Path(sys.argv[2])
while pid.exists():
 try:cmd=pid.read_bytes()
 except FileNotFoundError:break
 if b'collect_supermimic_static_perturbations.py' not in cmd:break
 time.sleep(10)
exitfile=root/'projects/supermimic_research/static_perturb_v1.exit'
for _ in range(30):
 if exitfile.exists():break
 time.sleep(1)
assert exitfile.read_text().strip()=='0'
p=root/'datasets/practice9/supermimic_tw_20261031/longterm_diagnostics/static_perturb_v1/results.json'
rows=json.loads(p.read_text())['rows'];assert len(rows)==16
assert len({(r['id'],r['mode']) for r in rows})==16
PY
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$STATE/static_perturb_v1" --checkpoint "$GPUFREE_DATA_ROOT/twrl_runs/supermimic_tw_20261031/physics_v3_1000/2026-10-31_05-15-56/checkpoint_final.pt" --output "$STATE/static_original_model.json"
python scripts/practice9/audit_supermimic_calibration_model.py --collection "$STATE/static_perturb_v1" --checkpoint "$STATE/wm_calibration_v1/world_model_calibrated.pt" --output "$STATE/static_calibrated_model.json"
python scripts/practice9/compare_supermimic_calibration.py --before "$STATE/static_original_model.json" --after "$STATE/static_calibrated_model.json" --output "$STATE/static_model_comparison.json"
