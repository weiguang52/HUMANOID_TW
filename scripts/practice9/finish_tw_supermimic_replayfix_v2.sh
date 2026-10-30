#!/usr/bin/env bash
# Queue after the known live pilot process; never restart training automatically.
set -euo pipefail
cd /root/gpufree-data/projects/HUMANOID_TW
source scripts/practice9/activate_supermimic_environment.sh
STATE="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/pilot_replayfix_v2_1700"
TRAIN_PID=${1:?Pass the verified live training PID}
exec 8>"$STATE/evaluation.lock"
flock -n 8 || { echo 'Evaluation worker already active' >&2; exit 1; }
printf '%s\n' waiting_for_training > "$STATE/evaluation_status"
trap 'printf "failed\n" > "$STATE/evaluation_status"' ERR
python - "$TRAIN_PID" <<'PY'
from pathlib import Path
import sys,time
pid=int(sys.argv[1]);path=Path(f'/proc/{pid}/cmdline')
while path.exists():
    try: command=path.read_bytes()
    except FileNotFoundError: break
    if b'supermimic.scripts.train_tw' not in command or b'pilot_replayfix_v2_1700' not in command:break
    time.sleep(15)
PY
[[ "$(cat "$STATE/status")" = trained_not_evaluated ]] || { echo 'Training did not finish successfully; inspect preserved checkpoints' >&2; exit 1; }
RUN="$GPUFREE_DATA_ROOT/twrl_runs/supermimic_tw_20261031/pilot_replayfix_v2_1700/2026-10-31_03-58-23"
TEST="$GPUFREE_DATA_ROOT/datasets/practice9/independent_holdout_20261025/test.json"
export TW_SUPERMIMIC_MJCF="$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.xml"
export MUJOCO_GL=egl
printf '%s\n' verifying_inputs > "$STATE/evaluation_status"
python - "$STATE" "$TEST" <<'PY'
from pathlib import Path
import json,sys,hashlib
root=Path('/root/gpufree-data');state,test_path=map(Path,sys.argv[1:]);train_path=root/'datasets/practice9/training_speed_blend_20261026/train.json'
train,test=[json.loads(p.read_text()) for p in (train_path,test_path)]
canonical=lambda x:x[1:] if x.startswith('M') else x
train_ids={canonical(r['id']) for r in train['motions']};test_ids={canonical(r['id']) for r in test['motions']}
assert not train_ids & test_ids
train_families={str(Path(r['source_family'])) for r in train['motions']}
test_families={str(Path(r['source_family'])) for r in test['motions']}
assert not train_families & test_families
assert len(test['motions'])==7
quality=json.loads((test_path.parent/'data_audit.json').read_text())
assert set(quality['accepted_ids'])=={r['id'] for r in test['motions']}
assert quality['test_manifest_sha256']==hashlib.sha256(test_path.read_bytes()).hexdigest()
report=dict(passed=True,train_clips=len(train['motions']),test_clips=len(test['motions']),mirror_normalized_id_overlap=sorted(train_ids & test_ids),source_family_overlap=sorted(train_families & test_families),test_ids=sorted(test_ids),original_selection_count=quality['selected'],rejected_ids=quality['not_accepted_ids'],train_manifest_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),test_manifest_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),note='Existing 12-source holdout yielded 7 accepted retargets; no rebalancing or replacement. Descriptions include mixed turning/bending/arm motions, not uniformly flat walking.')
(state/'evaluation_input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
PY
python scripts/practice9/audit_supermimic_training.py "$RUN/checkpoint_0001600.pt" "$RUN/checkpoint_final.pt" --output "$STATE/checkpoint_audit.json"
printf '%s\n' evaluating > "$STATE/evaluation_status"
python -u scripts/practice9/evaluate_supermimic_tw.py --manifest "$TEST" --checkpoint "$RUN/checkpoint_final.pt" --output "$STATE/evaluation" --max-steps 1000 --agents trained zero --video
python scripts/practice9/render_validation_references.py --manifest "$TEST" --output "$GPUFREE_DATA_ROOT/datasets/practice9/independent_holdout_20261025/source_videos"
python scripts/practice9/verify_supermimic_videos.py --evaluation "$STATE/evaluation" --source-index "$GPUFREE_DATA_ROOT/datasets/practice9/independent_holdout_20261025/source_videos/index.json" --palette "$GPUFREE_DATA_ROOT/datasets/practice9/supermimic_tw_20261031/robot/tw30_colored.colors.json"
printf '%s\n' evaluated_pending_visual_review > "$STATE/evaluation_status"
