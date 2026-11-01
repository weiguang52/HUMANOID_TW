"""Whole-sequence Newton foot projection constrained by original physical bounds."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from supermimic_grounded_qp import solve_grounded_qp


def retract(candidate, anchor, geometry, collision, mask, lo, hi, trust,
            velocity_matrix, original_delta, output):
    candidate = candidate.copy()
    active = np.repeat(mask, 12)
    scale = trust
    C = sp.diags(scale)
    records = []
    output = Path(output)
    for iteration in range(4):
        foot, _, J, _ = geometry(candidate)
        vectors = foot.reshape(len(mask), 2, 6)[mask]
        pe = float(np.linalg.norm(vectors[:, :, :3], axis=2).max())
        oe = float(np.linalg.norm(vectors[:, :, 3:], axis=2).max())
        if pe < 1e-8 and oe < 1e-7:
            return candidate, records, True
        heights, H = collision.evaluate(candidate)
        A = sp.vstack([sp.eye(len(candidate)), velocity_matrix, H, J[active]], format='csc')
        lower = np.r_[np.maximum(lo-candidate, anchor-trust-candidate),
                      -.06-original_delta-velocity_matrix@candidate,
                      -.0005+1e-6-heights, -foot[active]]
        upper = np.r_[np.minimum(hi-candidate, anchor+trust-candidate),
                      .06-original_delta-velocity_matrix@candidate,
                      np.full(len(heights), np.inf), -foot[active]]
        B = (A@C).tocsc()
        rows = 1/np.maximum(np.asarray(abs(B).max(axis=1).toarray()).ravel(), 1e-12)
        sol = solve_grounded_qp(2*sp.eye(len(candidate), format='csc'),
                                np.zeros(len(candidate)), (sp.diags(rows)@B).tocsc(),
                                rows*lower, rows*upper, 'clarabel')
        record = dict(iteration=iteration, status=sol.info.status,
                      solver_iterations=sol.info.iter, foot_position_before_m=pe,
                      foot_orientation_before_rad=oe)
        records.append(record)
        if sol.info.status_val != 1:
            output.mkdir(parents=True, exist_ok=True)
            sp.save_npz(output/f'failed_A_{iteration}.npz', A)
            np.savez_compressed(output/f'failed_vectors_{iteration}.npz', lower=lower,
                                upper=upper, candidate=candidate, scale=scale, rows=rows)
            return candidate, records, False
        delta = scale*sol.x
        dual = rows*sol.y
        objective_gradient = 2*delta/(scale*scale)
        kkt = float(np.linalg.norm(objective_gradient+A.T@dual, np.inf)/max(
            1, np.linalg.norm(objective_gradient, np.inf), np.linalg.norm(A.T@dual, np.inf)))
        violation = float(max(0, (lower-A@delta).max(), (A@delta-upper).max()))
        record.update(original_relative_stationarity=kkt,
                      unscaled_constraint_violation=violation,
                      correction_trust_units_max=float(abs(delta/scale).max()))
        if kkt > 1e-7 or violation > 1e-7:
            return candidate, records, False
        candidate += delta
    # Caller still applies original actual foot-vector, velocity, collision, force,
    # trust and objective-descent checks. Exhausting Newton steps is no acceptance.
    return candidate, records, True
