"""Export all eight audited corrected references for diagnostic replay only."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from supermimic_reference_reconstruction import reconstruct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Never overwrite diagnostic evidence'
    plans = {s: json.loads((a.state/f'{s}_plan.json').read_text()) for s in ('fit', 'validation')}
    source_path = Path(plans['fit']['source_plan'])
    assert source_path == Path(plans['validation']['source_plan'])
    source = json.loads(source_path.read_text())
    assert len(source['motions']) == 8 and len({x['source_family'] for x in source['motions']}) == 8
    evidence, corrections = {}, {}
    for split, plan in plans.items():
        expected = [x for x in source['motions'] if x['split'] == split]
        assert len(expected) == 4
        files = {name: a.state/f'{split}20'/f'{name}.json' for name in ('independent_audit', 'full_wrench_audit', 'full_circular_wrench')}
        files['float32'] = a.state/f'float32_{split}_roundtrip_audit.json'
        audits = {k: json.loads(v.read_text()) for k, v in files.items()}
        for k, audit in audits.items():
            assert [r['id'] for r in audit['rows']] == [x['id'] for x in expected]
            evidence[str(files[k])] = digest(files[k])
        for k in ('independent_audit', 'full_wrench_audit'):
            audit = audits[k]
            assert audit['complete_source_coverage'] and audit['asset_sha256'] == digest(a.asset)
            assert audit['plan_sha256'] == digest(a.state/f'{split}_plan.json')
        assert audits['independent_audit']['results_sha256'] == digest(a.state/f'{split}20/results.json')
        for item, geom, wrench, circle, dtype in zip(expected, *(audits[k]['rows'] for k in ('independent_audit', 'full_wrench_audit', 'full_circular_wrench', 'float32'))):
            cp = a.state/f'{split}20'/item['id']/'final_correction.npz'
            assert digest(item['file']) == item['sha256']
            assert all(r['correction_sha256'] == digest(cp) for r in (geom, wrench, circle, dtype))
            assert all(r['source_sha256'] == item['sha256'] for r in (geom, wrench, circle))
            assert geom['frames'] == dtype['frames'] == item['frames']
            assert geom['numerical_constraints_pass'] and geom['single_support_cop_outside'] == geom['double_support_cop_outside'] == 0
            assert wrench['known_frames'] == circle['known_frames'] == len(geom['wrench_samples'])
            assert wrench['optimistic_lp_infeasible'] == wrench['optimistic_lp_other_failure'] == circle['numerical_failures'] == 0
            assert dtype['collision_pass'] and dtype['speed_numerical_allowance_pass'] and dtype['float32_joint_bound_violation_rad'] <= 1e-7
            corrections[item['id']] = cp
    model = mujoco.MjModel.from_xml_path(str(a.asset))
    a.output.mkdir(parents=True)
    rows = []
    for item in source['motions']:
        with np.load(item['file']) as z:
            original = {k: z[k].copy() for k in z.files}
        with np.load(corrections[item['id']]) as z:
            delta = z['delta'].copy()
        names = original['joint_names'].tolist()
        lower = [i for i, name in enumerate(names) if name.startswith(('left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot'))]
        assert len(lower) == 14 and delta.shape == (item['frames'], 17)
        joints = original['joint_pos'].astype(float)
        joints[:, lower] += delta[:, 3:]
        rebuilt = reconstruct(original, model, joints, original['root_pos'].astype(float)+delta[:, :3])
        assert np.array_equal(rebuilt['joint_pos'][:, [i for i in range(30) if i not in lower]], original['joint_pos'][:, [i for i in range(30) if i not in lower]])
        out = a.output/f"{item['id']}.npz"
        np.savez_compressed(out, **rebuilt)
        rows.append(dict(item, file=str(out), sha256=digest(out), original_file=item['file'], original_sha256=item['sha256'], correction_sha256=digest(corrections[item['id']])))
    manifest = dict(source, motions=rows, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), diagnostic_only=True, training_export_approved=False, physical_acceptance=False, parent_plan=str(source_path), parent_plan_sha256=digest(source_path), asset_sha256=digest(a.asset), audit_sha256=evidence)
    (a.output/'collection_plan.json').write_text(json.dumps(manifest, indent=2))
    print('Eight diagnostic references written; independent exported-data audit required before replay')


if __name__ == '__main__':
    main()
