"""Explicit independent QP backend for whole-sequence geometric restoration.

Hash-binds the original restoration code. No source/iteration selection, no
physical threshold changes, and no training-reference export.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for name in ['plan', 'asset', 'output']:
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
assert plan['qp_solver'] == 'clarabel' and plan['numerics']['iterations'] == 20
assert plan['source_split'] in ['fit', 'validation']
assert not plan.get('first_source_only')
source = Path(__file__).with_name('restore_supermimic_sequence_stance.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == plan['restoration_source_sha256']
assert plan['physical_unit_scaling']
s = source.read_text()
old = "solver=osqp.OSQP();solver.setup(P=sp.triu(C@P@C/objective_scale,format='csc'),q=cs*gradient/objective_scale,A=(D@B).tocsc(),l=rs*l,u=rs*u,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True);sol=solver.solve()"
assert s.count(old) == 1
s = s.replace(old, "sol=__import__('supermimic_grounded_qp').solve_grounded_qp(sp.triu(C@P@C/objective_scale,format='csc'),cs*gradient/objective_scale,(D@B).tocsc(),rs*l,rs*u,backend='clarabel')")
# Original-unit primal residual gate is retained; explicitly enforce the same
# 1e-7 KKT audit used in the exact archived-QP diagnostic before applying steps.
old = '  if violation>1e-7:'
assert s.count(old) == 1
s = s.replace(old, "  if violation>1e-7 or entry['original_relative_stationarity']>1e-7:")
exec(compile(s, str(source)+'[explicit_clarabel_backend]', 'exec'),
     {'__name__': '__main__', '__file__': str(source)})
