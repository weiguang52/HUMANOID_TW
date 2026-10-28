"""Opt-in ankle yaw gain schedule driven only by reference planar speed."""
import torch
import math

def speed_gain(speed, maximum_gain=1.1):
    if not math.isfinite(maximum_gain) or maximum_gain<1:
        raise ValueError("Maximum damping gain must be finite and at least one")
    u=((speed-.05)/.05).clamp(0.,1.)
    return 1.+(maximum_gain-1.)*u*u*(3.-2.*u)

class LocomotionDamping:
    def __init__(self,env,names=None,maximum_gain=1.1):
        speed_gain(torch.zeros(1),maximum_gain)
        self.maximum_gain=float(maximum_gain)
        self.robot=env.scene['robot'];self.command=env.command_manager.get_term('motion')
        self.root=self.command.cfg.body_names.index('base_link')
        names=list(names) if names is not None else ['left_ankle_yaw_joint','right_ankle_yaw_joint']
        if len(names)!=2 or len(set(names))!=2:raise ValueError('Exactly two unique damping joints required')
        self.names=names
        self.ids=[self.robot.joint_names.index(n) for n in names]
        self.actuator=self.robot.actuators['leg_major']
        self.local_ids=[self.actuator.joint_names.index(n) for n in names]
        self.base=self.robot.data.joint_damping[:,self.ids].clone()
        if not torch.all(self.base==1.):raise ValueError('Schedule requires original selected joint damping 1.0')
        self.current=self.base.clone();self.records=[]
    def set(self,value):
        if not torch.equal(value,self.current):
            self.robot.write_joint_damping_to_sim(value,joint_ids=self.ids)
            self.actuator.damping[:,self.local_ids]=value
            self.current=value.clone()
    def reset(self):
        self.set(self.base);self.records=[]
    def step(self):
        speed=torch.linalg.vector_norm(self.command.body_lin_vel_w[:,self.root,:2],dim=-1)
        gain=speed_gain(speed,self.maximum_gain)
        self.set(self.base*gain[:,None])
        self.records.append(torch.stack([speed,gain],dim=-1).detach().cpu())
    def save(self,path):
        import numpy as np
        np.savez_compressed(path,reference_speed_and_gain=torch.stack(self.records).numpy(),joint_names=np.array(self.names),base_damping=self.base.cpu().numpy(),maximum_gain=self.maximum_gain)
