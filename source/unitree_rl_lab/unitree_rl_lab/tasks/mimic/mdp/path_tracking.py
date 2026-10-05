"""Horizontal reference progress, expressed in the robot yaw frame."""
import torch
from .size_tracking import normalized_pseudo_huber


def yaw_frame_xy(delta_xy, quat):
    w,x,y,z=quat.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    c,s=torch.cos(yaw),torch.sin(yaw)
    dx,dy=delta_xy.unbind(-1)
    return torch.stack((c*dx+s*dy,-s*dx+c*dy),-1)


def path_error_observation(env, command_name, scale):
    c=env.command_manager.get_term(command_name)
    delta=c.anchor_pos_w[:,:2]-c.robot_anchor_pos_w[:,:2]
    return (yaw_frame_xy(delta,c.robot_anchor_quat_w)/scale).clamp(-5,5)


def path_error_cost(env, command_name, scale):
    c=env.command_manager.get_term(command_name)
    return normalized_pseudo_huber(c.anchor_pos_w[:,:2]-c.robot_anchor_pos_w[:,:2],scale)
