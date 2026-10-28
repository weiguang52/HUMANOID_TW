"""Validate explicit batch transforms for checkpoints trained with leg filtering."""
from pathlib import Path
import json,math

def validate_jobs(leg,ankle,jobs):
    if leg['scope']!='legs14' or leg.get('mode')!='absolute':
        raise ValueError('Unsupported trained leg-filter contract')
    if leg.get('order')!='candidate legs then existing ankles':
        raise ValueError('Unsupported trained filter order')
    names=leg['joints']
    expected={side+'_'+joint for side in ['left','right'] for joint in ['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll']}
    if set(names)!=expected:raise ValueError('Saved leg joints differ from implemented filter')
    if len(names)!=14 or len(set(names))!=14 or not math.isfinite(leg['alpha']) or not 0<leg['alpha']<=1:
        raise ValueError('Invalid saved leg filter')
    if ankle['scope']!='ankles':raise ValueError('Expected trained outer ankle filter')
    if not jobs:raise ValueError('Trained leg filter requires explicit batch jobs')
    for j in jobs:
        if j.get('controller','policy')!='policy':raise ValueError('Trained leg-filter replay requires policy controller')
        if 'reference_speed_teacher_export' not in j:raise ValueError('Candidate leg filter needs explicit reference-speed blend')
        if 'blend_teacher_export' in j or 'candidate_hip_filter_alpha' in j:raise ValueError('Conflicting candidate transform')
        if j.get('candidate_leg_filter_alpha')!=leg['alpha'] or j.get('candidate_leg_filter_mode','absolute')!='absolute':
            raise ValueError('Batch must preserve trained candidate leg filter')
        if j.get('leg_filter_alpha')!=ankle['alpha'] or j.get('leg_filter_scope')!='ankles' or j.get('leg_filter_mode','absolute')!='absolute':
            raise ValueError('Batch must preserve trained outer ankle filter')
    return {'leg':leg,'ankle':ankle,'matched_batch_jobs':len(jobs),'order':'candidate legs then existing ankles'}

def validate_checkpoint_jobs(checkpoint,jobs):
    params=Path(checkpoint).parent/'params';path=params/'leg_action_filter.json'
    if not path.exists():return None
    ankle=json.loads((params/'action_filter.json').read_text())
    return validate_jobs(json.loads(path.read_text()),ankle,jobs)
