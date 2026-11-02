"""Independently audit full native-state/warmstart replay pairs; no fitting."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection', type=Path, required=True)
p.add_argument('--native', type=Path, required=True)
p.add_argument('--warmstart', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), 'Preserve previous audit evidence'
source = json.loads((a.collection / 'results.json').read_text())['rows']
assert len(source) == 32
assert len({r['id'] for r in source}) == 8
assert len({(r['id'], r['frame']) for r in source}) == 32
summary = {}
plans = []
for arm, folder in [('native', a.native), ('warmstart', a.warmstart)]:
    plan = json.loads((folder / 'plan.json').read_text())
    results = json.loads((folder / 'results.json').read_text())
    assert plan['phase'] == 'cold' and plan['cold_all_frames']
    assert plan['restore_native_coordinates'] and not plan['clear_native_transients']
    assert plan['restore_native_warmstart'] == (arm == 'warmstart')
    assert plan['native_replay_atol'] == 1e-5
    assert len(results['rows']) == 32
    assert [r['file'] for r in results['rows']] == [r['trace'] for r in source]
    assert [r['file'] for r in plan['source_files']] == [r['trace'] for r in source]
    plans.append(plan)
    failures = []
    points = 0
    max_error = 0.0
    for row, replay, bound in zip(source, results['rows'], plan['source_files']):
        src = a.collection / row['trace']
        assert src.resolve().is_relative_to(a.collection.resolve())
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        assert sha == row['sha256'] == bound['sha256']
        with np.load(src, allow_pickle=False) as z:
            original = {k: z[k].copy() for k in z.files}
        n = len(original['state'])
        assert n == row['steps'] == replay['source_frames'] == replay['replayed_frames']
        for before, after in [('state', 'next_state'), ('native_qpos', 'native_next_qpos'), ('native_qvel', 'native_next_qvel')]:
            assert np.array_equal(original[after][:-1], original[before][1:])
        assert original['native_qacc_warmstart'].shape == (n, 36)
        assert np.isfinite(original['native_qacc_warmstart']).all()
        path = folder / replay['trace']
        assert path.resolve().is_relative_to(folder.resolve())
        with np.load(path, allow_pickle=False) as z:
            actual = {k: z[k].copy() for k in z.files}
        assert np.array_equal(actual['index'], np.arange(n))
        for saved, expected in [('restored_state', 'state'), ('context', 'context'), ('recorded_next_state', 'next_state'), ('native_restored_qpos', 'native_qpos'), ('native_restored_qvel', 'native_qvel')]:
            assert np.array_equal(actual[saved], original[expected]), (arm, row['trace'], saved)
        if arm == 'warmstart':
            assert np.array_equal(actual['native_restored_qacc_warmstart'], original['native_qacc_warmstart'])
        assert actual['actual_next_state'].shape == (n, 73)
        assert all(np.isfinite(v).all() for v in actual.values())
        error = np.abs(actual['actual_next_state'] - original['next_state'])
        assert float(error.max()) == replay['max_next_state_error']
        for index in np.flatnonzero(error.max(axis=1) > 1e-5):
            coordinate = int(error[index].argmax())
            failures.append(dict(trace=row['trace'], index=int(index), coordinate=coordinate, error=float(error[index, coordinate])))
        points += n
        max_error = max(max_error, float(error.max()))
    summary[arm] = dict(traces=32, points=points, failed_points=len(failures), max_next_state_error=max_error, failures=failures, all_pass=not failures)
    assert results['all_next_states_within_native_atol'] == (not failures)
# Plans may differ only in the intended warmstart intervention.
left, right = [dict(x) for x in plans]
left.pop('restore_native_warmstart'); right.pop('restore_native_warmstart')
assert left == right
keys = lambda arm: {(x['trace'], x['index']) for x in summary[arm]['failures']}
report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), coverage_verified=True, native_state_exact=True, warmstart_restoration_exact=True, unchanged_threshold=1e-5, arms=summary, common_failures=len(keys('native') & keys('warmstart')), native_only=len(keys('native') - keys('warmstart')), warmstart_only=len(keys('warmstart') - keys('native')), scope='Full paired native replay diagnostic only; no model or policy acceptance')
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: {j: v for j, v in value.items() if j != 'failures'} for k, value in summary.items()}))
