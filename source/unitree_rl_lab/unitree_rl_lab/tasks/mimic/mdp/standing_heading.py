"""Opt-in reference heading control for stationary standing clips."""
import math
import torch


def heading_cost(reference, actual, reference_xy, reference_yaw_rate, standing, scale=0.35):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite angular scale required')
    def yaw(q):
        w,x,y,z=q.unbind(-1)
        return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    delta=yaw(reference)-yaw(actual)
    error=torch.atan2(torch.sin(delta),torch.cos(delta))
    gate=((0.025-reference_xy.norm(dim=-1))/0.02).clamp(0,1)
    gate=gate*((0.3-reference_yaw_rate.abs())/0.2).clamp(0,1)*standing.bool()
    return (torch.sqrt(1+(error/scale).square())-1)*gate


def stationary_heading(env,command_name,scale=0.35):
    from .stability_tracking import standing_mask
    c=env.command_manager.get_term(command_name)
    i=c.cfg.body_names.index('base_link')
    return heading_cost(c.body_quat_w[:,i],c.robot_body_quat_w[:,i],
                        c.body_lin_vel_w[:,i,:2],c.body_ang_vel_w[:,i,2],
                        standing_mask(c),scale)
