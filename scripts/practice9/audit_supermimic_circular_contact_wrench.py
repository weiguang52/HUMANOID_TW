"""Circular-friction necessary feasibility of archived ground-projected wrenches.

Does not demonstrate actual contacts, actuator feasibility, or stable control.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import clarabel
import numpy as np
from scipy import sparse


def cross(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def audit(sample):
    points, coefficients = [], []
    for geom in sample['geometry']:
        p = np.array(geom['bottom_corners_w_m'], dtype=float)
        p[:, 2] = 0  # Same optimistic projected support as parent diagnostic.
        points.extend(p)
        coefficients.extend([geom['asset_sliding_friction']] * len(p))
    count = len(points)
    equilibrium = np.vstack([np.tile(np.eye(3), (1, count)),
                             np.hstack([cross(p) for p in points])])
    required = np.r_[sample['required_force_n'], sample['required_torque_origin_nm']]
    blocks = [equilibrium]
    cones = [clarabel.ZeroConeT(6)]
    for i, mu in enumerate(coefficients):
        transform = np.zeros((3, 3 * count))
        transform[:, 3*i:3*i+3] = [[0, 0, mu], [1, 0, 0], [0, 1, 0]]
        blocks.append(-transform)
        cones.append(clarabel.SecondOrderConeT(3))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solution = clarabel.DefaultSolver(
        sparse.eye(3*count, format='csc'), np.zeros(3*count),
        sparse.csc_matrix(np.vstack(blocks)),
        np.r_[required, np.zeros(3*count)], cones, settings).solve()
    result = dict(frame=sample['frame'], status=str(solution.status), iterations=solution.iterations)
    if str(solution.status) == 'Solved':
        forces = np.array(solution.x).reshape(count, 3)
        residual = float(abs(equilibrium @ np.array(solution.x) - required).max())
        violation = float(max(0, (np.linalg.norm(forces[:, :2], axis=1)
                                 - np.array(coefficients)*forces[:, 2]).max(),
                              -forces[:, 2].min()))
        result.update(equilibrium_residual=residual, circular_cone_violation_n=violation,
                      corner_forces_n=forces.tolist(),
                      numerical_feasible=residual <= 1e-7 and violation <= 1e-7)
    else:
        result['numerical_feasible'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rows, hashes = [], {}
    seen = set()
    for path in args.inputs:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        parent = json.loads(raw)
        for row in parent['rows']:
            key = (row['id'], row['arm'])
            assert key not in seen
            seen.add(key)
            samples = [audit(s) for s in row['samples']]
            rows.append(dict(id=row['id'], arm=row['arm'],
                             source_sha256=row['source_sha256'],
                             correction_sha256=row['correction_sha256'],
                             known_frames=len(samples),
                             numerical_failures=sum(not s['numerical_feasible'] for s in samples),
                             samples=samples))
    result = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Archived full wrench, circular cone, optimistic ground-projected corners. '
                        'Necessary feasibility only: no actual contact, joint torque, PD tracking or policy acceptance.',
                  input_sha256=hashes, rows=rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='samples'} for row in rows]))


if __name__ == '__main__':
    main()
