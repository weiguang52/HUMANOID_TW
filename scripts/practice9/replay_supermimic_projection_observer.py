"""Read-only native substep observer; no physics or tolerance changes."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import numpy as np
import torch
import warp as wp
from supermimic.config.tw import velocity_limits as limits

assert '--restore-native-coordinates' in sys.argv and '--restore-native-warmstart' in sys.argv
output = Path(sys.argv[sys.argv.index('--output') + 1])
assert not output.exists()
original_step = limits._project_registered_velocity
original_project = limits.project_velocity_mass_metric
records = []
current = None


def observe_project(velocity, mass, caps, *args, **kwargs):
    assert current is not None
    before = velocity.clone()
    inertia = mass.clone()
    value, stats = original_project(velocity, mass, caps, *args, **kwargs)
    assert torch.equal(before, velocity) and torch.equal(inertia, mass), 'Projection modified inputs'
    current.update(active=True, iterations=stats['iterations'], residual=stats['residual'], momentum_error=stats['momentum_error'], energy_increase=stats['energy_increase'])
    current['dense_mass'] = inertia[0].cpu().numpy().copy()
    return value, stats


def observe_step(model, data):
    global current
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream())):
        pre = wp.to_torch(data.qvel).clone()
        qpos = wp.to_torch(data.qpos).clone()
        sparse_mass = wp.to_torch(data.M).clone()
        current = dict(active=False, iterations=0, residual=0., momentum_error=0., energy_increase=0.)
        original_step(model, data)
        post = wp.to_torch(data.qvel).clone()
        assert torch.equal(qpos, wp.to_torch(data.qpos)) and torch.equal(sparse_mass, wp.to_torch(data.M))
        if not current['active']:
            assert torch.equal(pre, post)
            current['dense_mass'] = np.zeros((36, 36), dtype=np.float32)
        current.update(pre_velocity=pre[0].cpu().numpy().copy(), post_velocity=post[0].cpu().numpy().copy(), qpos=qpos[0].cpu().numpy().copy(), sparse_mass=sparse_mass[0].cpu().numpy().copy())
        records.append(current)
        current = None


limits._project_registered_velocity = observe_step
limits.project_velocity_mass_metric = observe_project
try:
    runpy.run_path(str(Path(__file__).with_name('replay_supermimic_motion_pd.py')), run_name='__main__')
finally:
    limits._project_registered_velocity = original_step
    limits.project_velocity_mass_metric = original_project
    if output.exists() and records:
        arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(output / 'projection_substeps.npz', **arrays)
        report = dict(diagnostic_only=True, physics_changed=False, input_mutation_checks_passed=True, substeps=len(records), active_substeps=int(arrays['active'].sum()), observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), physics_source_sha256=hashlib.sha256(Path(limits.__file__).read_bytes()).hexdigest(), scope='Read-only synchronization may affect numeric execution timing; no training/policy acceptance')
        (output / 'projection_observer.json').write_text(json.dumps(report, indent=2))
