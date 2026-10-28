"""Optional teacher/student action partition; not enabled without explicit scope.

Only names and current policy actions are used. No motion IDs, future samples,
reference corrections, filtering, physics changes or persistent state.
"""
import torch
from reference_speed_blend import mix_actions

LEG_NAMES = tuple(side + '_' + joint for side in ('left', 'right')
    for joint in ('hip_pitch_joint', 'hip_roll_joint', 'hip_yaw_joint',
                  'knee_pitch_joint', 'ankle_yaw_joint', 'ankle_pitch_joint', 'foot_roll'))
WAIST_NAMES = ('waist_yaw', 'waist_roll_joint', 'waist_pitch_joint')
ARM_NAMES = tuple(side + '_' + joint for side in ('left', 'right')
    for joint in ('shoulder_pitch_joint', 'shoulder_roll_joint',
                  'shoulder_yaw_joint', 'elbow_pitch_joint', 'wrist_pitch'))
HEAD_NAMES = ('neck', 'neck_linkage_roll', 'head_pitch')
EXPECTED_NAMES = LEG_NAMES + WAIST_NAMES + ARM_NAMES + HEAD_NAMES


def make_scope_contract(names, scope='all'):
    if scope not in ('all', 'lower_body_waist'):
        raise ValueError('Unknown candidate action scope')
    try:
        names = list(names)
    except TypeError as error:
        raise ValueError('Expected an ordered joint-name sequence') from error
    if len(names) != 30 or any(not isinstance(n, str) for n in names) or len(set(names)) != 30 or set(names) != set(EXPECTED_NAMES):
        raise ValueError('Expected exact unique30 joint action names')
    selected = set(EXPECTED_NAMES if scope == 'all' else LEG_NAMES + WAIST_NAMES)
    return {'version': 1, 'scope': scope, 'action_joint_names': names,
            'candidate_joint_names': [n for n in names if n in selected],
            'teacher_joint_names': [n for n in names if n not in selected]}


def mix_scoped_actions(teacher, candidate, weight, scope='all', action_names=None):
    # Explicit legacy bypass preserves original math, views and RNG exactly.
    if scope == 'all':
        return mix_actions(teacher, candidate, weight)
    if action_names is None:
        raise ValueError('Scoped mixture requires current joint action order')
    contract = make_scope_contract(action_names, scope)
    if teacher.shape != candidate.shape or teacher.ndim < 2 or teacher.shape[-1] != 30:
        raise ValueError('Scoped mixture requires matching30 action tensors')
    if weight.shape != teacher.shape[:-1] or not torch.isfinite(weight).all() or bool(((weight < 0) | (weight > 1)).any()):
        raise ValueError('Scoped mixture requires finite per-environment weights in[0,1]')
    output = mix_actions(teacher, candidate, weight).clone()
    frozen = torch.tensor([n in contract['teacher_joint_names'] for n in contract['action_joint_names']],
                          dtype=torch.bool, device=teacher.device)
    output[..., frozen] = teacher[..., frozen]
    return output
