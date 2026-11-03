"""Verify frozen-fit common-gradient diagnostics against the previous five directions.

This report is optimizer feasibility evidence, never a policy acceptance gate.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--diagnostic', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
args = parser.parse_args()
report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Frozen fit batches only; no validation or policy acceptance',
          'policy_acceptance': False, 'arms': {}}
for arm in ['control', 'candidate']:
    path = args.diagnostic / arm / 'audit.json'
    reference_path = args.reference / arm / 'audit.json'
    current = json.loads(path.read_text())
    previous = json.loads(reference_path.read_text())
    assert len(current['rows']) == len(previous['rows']) == 16
    assert current['source_weights_unchanged'] and current['source_optimizer_steps'] == 0
    assert current['plan']['checkpoint_sha256'] == previous['plan']['checkpoint_sha256']
    checkpoint = Path(current['plan']['checkpoint'])
    with checkpoint.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == current['plan']['checkpoint_sha256']
    counts = {name: 0 for name in ['nonzero_static_full_batches', 'nonzero_all_five_batches', 'first_order_all_five_descent_batches']}
    rows = []
    for new, old in zip(current['rows'], previous['rows']):
        for key in ['batch', 'static_loss', 'full_loss', 'gradient_cosine', 'component_baseline']:
            assert new[key] == old[key], (arm, new['batch'], key)
        for name, direction in old['directions'].items():
            assert new['directions'][name] == direction, (arm, new['batch'], name)
        weights = new['response_common_weights']
        assert len(weights) == 5 and all(math.isfinite(w) and w >= 0 for w in weights)
        assert abs(sum(weights) - 1) < 1e-12
        slopes = new['response_common_normalized_slopes']
        counts['first_order_all_five_descent_batches'] += all(s < 0 for s in slopes)
        probes = new['directions']['response_common_min_norm']['probes']
        assert [p['factor'] for p in probes] == [1., .5, .25, .125, .0625, .03125, .015625, .0078125, 0.]
        baseline = new['component_baseline']
        identity = probes[-1]
        assert identity['loss'] == new['static_loss']
        assert abs(identity['full_loss'] - new['full_loss']) <= 1e-6 * max(1, abs(new['full_loss']))
        assert all(abs(identity['component_losses'][key] - value) <= 1e-6 * max(1, abs(value)) for key, value in baseline.items())
        eligible = []
        for probe in probes[:-1]:
            assert math.isfinite(probe['loss']) and math.isfinite(probe['full_loss'])
            assert all(math.isfinite(v) for v in probe['component_losses'].values())
            both = probe['loss'] <= new['static_loss'] and probe['full_loss'] <= new['full_loss']
            assert both == probe['both_nonincrease']
            if both:
                eligible.append(probe)
        protected = [p['factor'] for p in eligible if all(p['component_losses'][key] <= baseline[key] for key in ['response_value', 'root_state_value', 'root_pose_value'])]
        counts['nonzero_static_full_batches'] += bool(eligible)
        counts['nonzero_all_five_batches'] += bool(protected)
        rows.append({'batch': new['batch'], 'all_five_nonincrease_factors': protected})
    report['arms'][arm] = {'checkpoint_sha256': current['plan']['checkpoint_sha256'],
        'original_five_directions_exact': True, 'identity_replay_passed': True,
        'counts': counts, 'rows': rows}
output = args.diagnostic / 'independent_review.json'
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
