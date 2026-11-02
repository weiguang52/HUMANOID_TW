"""Explicit diagnostic wrapper; does not modify installed solver files."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
from supermimic_sparse_reduction_diagnostic import install_diagnostic, SOLVER_SHA256

if '--output' not in sys.argv:
    raise SystemExit('Replay --output is required')
output = Path(sys.argv[sys.argv.index('--output') + 1])
assert not output.exists()
original, calls = install_diagnostic()
try:
    runpy.run_path(str(Path(__file__).with_name('replay_supermimic_motion_pd.py')), run_name='__main__')
finally:
    from mujoco_warp._src import solver
    solver._update_constraint_init_qfrc_constraint_sparse = original
    if output.exists():
        record = dict(diagnostic_only=True, training_approved=False, promotion_approved=False, solver_source_sha256=SOLVER_SHA256, wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), reduction_sha256=hashlib.sha256(Path(__file__).with_name('supermimic_sparse_reduction_diagnostic.py').read_bytes()).hexdigest(), **calls)
        (output / 'fixed_reduction_diagnostic.json').write_text(json.dumps(record, indent=2))
assert calls['factory_calls'] > 0, 'Diagnostic kernel was never selected'
