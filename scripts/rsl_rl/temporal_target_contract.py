"""Explicit, backward-compatible target of training-only temporal losses."""
def validate_target(coefficient,order,target):
    if target not in ('teacher_relative','student_mean'):
        raise ValueError('Unknown temporal target')
    if target=='student_mean' and (order!=2 or coefficient<=0):
        raise ValueError('Student mean curvature requires order2 and positive coefficient')

def validate_resume_target(saved,requested):
    if saved.get('temporal_coefficient',0) and saved.get('temporal_target','teacher_relative')!=requested:
        raise ValueError('Resume must preserve checkpoint temporal target')
