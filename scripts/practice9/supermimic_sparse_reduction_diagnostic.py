"""Opt-in process-local fixed-order sparse constraint-force diagnostic.

Keeps J.T @ force, float32 arithmetic, skip rules and compact mapping. Changes
only summation order for one solver kernel. Never enable for training implicitly.
"""
from pathlib import Path
import hashlib
import warp as wp

SOLVER_SHA256 = 'f07a7c33e038a312ea5c8cab4ec718b24d5e7d163dc8435a592118c9fc0e07fb'


def make_fixed_order_kernel(compact):
    COMPACT = bool(compact)

    @wp.kernel(module='unique', enable_backward=False)
    def kernel(
        nefc: wp.array[int],
        rownnz: wp.array2d[int],
        rowadr: wp.array2d[int],
        colind: wp.array3d[int],
        jacobian: wp.array3d[float],
        force: wp.array2d[float],
        dof_cdof: wp.array2d[int],
        changed: wp.array[int],
        done: wp.array[bool],
        result: wp.array2d[float],
    ):
        world, dof = wp.tid()
        if dof >= result.shape[1] or done[world] or changed[world] == 0:
            return
        total = float(0.0)
        for row in range(nefc[world]):
            value = force[world, row]
            if value != 0.0:
                for item in range(rownnz[world, row]):
                    offset = rowadr[world, row] + item
                    column = colind[world, 0, offset]
                    if wp.static(COMPACT):
                        column = dof_cdof[world, column]
                    if column == dof:
                        total += jacobian[world, 0, offset] * value
        result[world, dof] = total

    return kernel


def install_diagnostic():
    from mujoco_warp._src import solver
    path = Path(solver.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOLVER_SHA256:
        raise RuntimeError('Unsupported solver source: ' + actual)
    original = solver._update_constraint_init_qfrc_constraint_sparse
    kernels = {flag: make_fixed_order_kernel(flag) for flag in (False, True)}
    calls = {'factory_calls': 0}

    def replacement(compact):
        calls['factory_calls'] += 1
        return kernels[bool(compact)]

    solver._update_constraint_init_qfrc_constraint_sparse = replacement
    return original, calls
