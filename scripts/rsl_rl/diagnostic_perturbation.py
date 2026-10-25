"""Audited instantaneous horizontal root-velocity perturbations (not force pulses)."""
import math,random
import torch

def vector(seed,speed):
    if not math.isfinite(speed) or not 0<=speed<=0.1:
        raise ValueError('Audited speeds are finite and within [0,0.1] m/s')
    angle=random.Random(int(seed)).uniform(-math.pi,math.pi)
    return [speed*math.cos(angle),speed*math.sin(angle)]

def validate(jobs,args):
    if not args.standing_diagnostic:raise ValueError('Perturbation requires diagnostic mode')
    for job in jobs:
        vector(args.seed,float(job['push_speed']))
        if int(job.get('push_step',250))!=250 or int(job['steps'])!=1000:
            raise ValueError('Audited protocol: kick at5s, horizon20s')

def apply(env,job,step,seed):
    if step!=int(job.get('push_step',250)):return None
    robot=env.scene['robot']
    before=robot.root_physx_view.get_velocities().clone()
    target=before.clone()
    delta=torch.tensor(vector(seed,float(job['push_speed'])),device=target.device,dtype=target.dtype)
    target[:,:2]+=delta
    if float(job['push_speed'])>0:robot.write_root_velocity_to_sim(target)
    actual=robot.root_physx_view.get_velocities().clone()
    error=float((actual-target).abs().max().item())
    if error>1e-6:raise RuntimeError(f'Physical velocity write mismatch {error}')
    return {'step':step,'time_s':step*env.step_dt,'requested_delta_xy':delta.cpu().tolist(),
            'before':before[0].cpu().tolist(),'target':target[0].cpu().tolist(),
            'physics_readback':actual[0].cpu().tolist(),'max_write_error':error}

def after_step(env,record):
    record['after_control_step']=env.scene['robot'].root_physx_view.get_velocities()[0].cpu().tolist()
