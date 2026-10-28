"""Opt-in release-only reference-speed handoff; no additional policy inputs."""
import numpy as np
import torch
from release_gate import TorchReleaseGate
from reference_speed_blend import candidate_weight,mix_actions

class ReleaseSpeedBlend:
    def __init__(self,base,tau,dt):
        if base.hip_filter is not None or base.leg_filter is not None:
            raise ValueError('Release experiment cannot combine candidate hip/leg filters')
        self.base=base;self.gate=TorchReleaseGate(tau,dt)
        self.trace=[];self.released=[]
        self.audit=dict(base.audit)
        self.audit['release_gate']={'tau_seconds':float(tau),'control_dt':float(dt),
            'rule':'max(raw_weight, previous_weight * exp(-dt/tau)); tau0 identity',
            'reset':'clear per environment after done','initial':'raw weight'}
    @torch.inference_mode()
    def __call__(self,obs,action):
        b=self.base
        speed=torch.linalg.vector_norm(b.command.body_lin_vel_w[:,b.index,:2],dim=-1)
        raw=candidate_weight(speed,b.maximum,b.low,b.high)
        weight=self.gate.step(raw)
        key='contact_base_policy' if 'contact_base_policy' in obs else 'policy'
        self.audit['observation_key']=key
        teacher=b.base.teacher(obs[key])
        self.trace.append(torch.stack([speed,raw],dim=-1).cpu().numpy())
        self.released.append(weight.cpu().numpy().copy())
        return mix_actions(teacher,action,weight)
    def reset(self,dones):
        self.base.reset(dones);self.gate.reset(dones)
    def save_trace(self,path):
        np.savez_compressed(path,speed_and_candidate_weight=np.asarray(self.trace),
                            released_candidate_weight=np.asarray(self.released))
