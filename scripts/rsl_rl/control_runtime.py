"""Persist custom action/actuator settings alongside each checkpoint run."""
import json
import os
from pathlib import Path


def snapshot(cfg):
    action=getattr(cfg.actions,'JointPositionAction',None)
    if action is None or not hasattr(action,'smoothing_tau'): return None
    return dict(version=1,target_velocity_scale=action.target_velocity_scale,smoothing_tau=action.smoothing_tau,limit_target_velocity=action.limit_target_velocity,
        actuators={name:dict(stiffness=a.stiffness,damping=a.damping) for name,a in cfg.scene.robot.actuators.items()})


def save(cfg,run_dir):
    payload=snapshot(cfg)
    if payload is not None:
        path=Path(run_dir)/'params/control_runtime.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2)+'\n')


def restore(cfg,checkpoint):
    path=Path(checkpoint).parent/'params/control_runtime.json'
    if not path.exists(): return
    payload=json.loads(path.read_text())
    if payload['version']!=1: raise ValueError('Unsupported control runtime version')
    action=cfg.actions.JointPositionAction
    for field,key in [('target_velocity_scale','P9_TARGET_VELOCITY_SCALE'),('smoothing_tau','P9_TARGET_SMOOTHING_TAU'),('limit_target_velocity','P9_TARGET_LIMIT_VELOCITY')]:
        if key not in os.environ: setattr(action,field,payload[field])
    for name,values in payload['actuators'].items():
        for field,key in [('stiffness','P9_PD_STIFFNESS_MULT'),('damping','P9_PD_DAMPING_MULT')]:
            if key not in os.environ: setattr(cfg.scene.robot.actuators[name],field,values[field])
