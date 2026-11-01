"""Run all original validation sources with the hash-bound frozen fit method.

Only changes source coverage assertions in the archived instrumentation wrapper;
solver, objective, numerical budget and acceptance gates remain hash-bound.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
for key in ['plan', 'asset', 'output']:
    p.add_argument('--'+key, type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
assert plan['source_split'] == 'validation' and plan['all_validation_sources']
assert plan['all_fit_sources']  # Legacy wrapper flag means all selected sources.
assert not plan.get('replay_source_id')
contract_path = Path(plan['frozen_fit_contract'])
assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == plan['frozen_fit_contract_sha256']
contract = json.loads(contract_path.read_text())
fit_path = Path(plan['frozen_fit_plan'])
assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == contract['fit_plan_sha256']
fit = json.loads(fit_path.read_text())
for key in ['source_plan', 'source_plan_sha256', 'phase_review', 'phase_review_sha256',
            'scales', 'numerics', 'restoration_contract', 'restoration_prior_weight',
            'collision_constraints', 'tracking_objective', 'dynamic_contract',
            'dynamic_prior_weight', 'qp_solver', 'solver_source_sha256',
            'strict_linear_foot_equalities', 'coupled_foot_retraction']:
    assert plan[key] == fit[key], ('Frozen fit parameter changed', key)
for name, expected in contract['source_sha256'].items():
    archived = contract_path.parent/name
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected
    if name != 'replay_supermimic_grounded_line_search.py':
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
sources = json.loads(Path(plan['source_plan']).read_text())['motions']
validation_ids = [item['id'] for item in sources if item['split'] == 'validation']
assert validation_ids == contract['validation_ids']
assert set(plan['initial_correction_sha256']) == set(validation_ids)
wrapper = contract_path.parent/'replay_supermimic_grounded_line_search.py'
s = wrapper.read_text()
old = "assert plan['source_split']=='fit'"
assert s.count(old) == 1
s = s.replace(old, "assert plan['source_split']=='validation'")
old = "if item['split']=='fit'"
assert s.count(old) == 1
s = s.replace(old, "if item['split']=='validation'")
# Existing wrapper reads arguments again; sibling solver remains original/hash-bound.
exec(compile(s, str(wrapper)+'[all_validation_coverage_only]', 'exec'),
     {'__name__': '__main__', '__file__': str(Path(__file__).with_name(wrapper.name))})
