"""Isolated schema/compatibility checks for an explicitly scoped two-policy run.

No training or playback entry point imports this module yet. Legacy contracts
remain version1 and are returned unchanged when scope='all'.
"""
from candidate_action_scope import make_scope_contract


def validate_scope_contract(value):
    if not isinstance(value, dict) or set(value) != {'version', 'scope',
            'action_joint_names', 'candidate_joint_names', 'teacher_joint_names'}:
        raise ValueError('Malformed action scope contract')
    if value['version'] != 1:
        raise ValueError('Unsupported action scope version')
    expected = make_scope_contract(value['action_joint_names'], value['scope'])
    if value != expected:
        raise ValueError('Action scope membership/order differs from declared names')
    return expected


def add_training_scope(base, names, scope='all'):
    if not isinstance(base, dict) or base.get('version') != 1 or 'candidate_action_scope' in base:
        raise ValueError('Expected an unscoped legacy training mixture contract')
    if scope == 'all':
        return base
    action_scope = make_scope_contract(names, scope)
    result = dict(base)
    result.update(version=2, candidate_action_scope=action_scope)
    return result


def split_training_scope(contract):
    if not isinstance(contract, dict):
        raise ValueError('Expected training mixture contract')
    if contract.get('version') == 3:
        from controlled_action_likelihood import detach_training_contract
        physical, likelihood = detach_training_contract(contract)
        return split_training_scope(physical)
    if contract.get('version') == 1:
        if 'candidate_action_scope' in contract:
            raise ValueError('Legacy version1 cannot silently carry an action scope')
        return contract, None
    if contract.get('version') != 2 or 'candidate_action_scope' not in contract:
        raise ValueError('Unsupported scoped training mixture version')
    scope = validate_scope_contract(contract['candidate_action_scope'])
    if scope['scope'] != 'lower_body_waist':
        raise ValueError('Version2 requires explicit lower_body_waist scope')
    base = dict(contract)
    base.pop('candidate_action_scope')
    base['version'] = 1
    return base, scope


def validate_scope_jobs(contract, jobs):
    """Only partition compatibility; caller must also validate original base contract."""
    base, scope = split_training_scope(contract)
    if not jobs:
        raise ValueError('Scoped contract requires explicit evaluation jobs')
    expected = scope['scope'] if scope is not None else 'all'
    for job in jobs:
        if job.get('candidate_action_scope', 'all') != expected:
            raise ValueError('Replay action scope differs from trained checkpoint')
    return base, scope


def require_scope_action_names(contract, actual_names):
    base, scope = split_training_scope(contract)
    if scope is None:
        return None
    actual = make_scope_contract(actual_names, scope['scope'])
    if actual != scope:
        raise ValueError('Runtime joint action order differs from saved action scope')
    return actual


def validate_scope_resume(saved_scope, current_scope, allow_legacy_initialization=False):
    current = validate_scope_contract(current_scope)
    if saved_scope is None:
        if current['scope'] != 'all' and not allow_legacy_initialization:
            raise ValueError('Scoped run requires explicit legacy initialization flag')
        return {'legacy_initialization': True, 'scope': current['scope']}
    saved = validate_scope_contract(saved_scope)
    if saved != current:
        raise ValueError('Resume must preserve exact saved action partition and joint order')
    return {'legacy_initialization': False, 'scope': current['scope']}
