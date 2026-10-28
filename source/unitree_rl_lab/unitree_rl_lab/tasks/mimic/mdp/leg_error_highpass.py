"""Causal leg tracking-error high-pass state; no policy input changes."""
import math
import torch

class LegErrorHighpass:
    def __init__(self, cutoff_hz, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
        if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < .5/dt:
            raise ValueError('Cutoff must lie between zero and Nyquist')
        self.decay=math.exp(-2*math.pi*cutoff_hz*dt)
        self.low=None

    def update(self, error, reset):
        if error.ndim != 2 or reset.shape != error.shape[:1] or reset.dtype != torch.bool:
            raise ValueError('Expected environment-by-joint error and boolean per-environment reset')
        if self.low is None:
            self.low=error.detach().clone()
        if self.low.shape != error.shape:
            raise ValueError('Environment or joint shape changed')
        low=self.decay*self.low+(1-self.decay)*error
        low=torch.where(reset[:,None],error,low)
        result=error-low
        self.low=low.detach()
        return result


def moving_leg_highpass_error(env, command_name='motion', cutoff_hz=8.0, scale=.01, filter_order=1, trace_steps=0):
    if type(filter_order) is not int or filter_order not in (1,4):
        raise ValueError('Highpass order must be 1 or 4')
    if type(trace_steps) is not int or trace_steps<0:
        raise ValueError('Nonnegative integer trace length required')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite angular scale required')
    command=env.command_manager.get_term(command_name)
    if not hasattr(command, '_highpass_leg_ids'):
        names=[command.robot.joint_names[i] for i in command.joint_indexes.tolist()]
        expected=[s+'_'+j+'_joint' for s in ['left','right'] for j in
                  ['hip_pitch','hip_roll','hip_yaw','knee_pitch','ankle_yaw','ankle_pitch']]
        if any(names.count(n)!=1 for n in expected):
            raise ValueError('Expected twelve unique leg joints in motion order')
        command._highpass_leg_ids=[names.index(n) for n in expected]
        if filter_order==1:
            command._leg_error_highpass=LegErrorHighpass(cutoff_hz,env.step_dt)
        else:
            from .sos_error_highpass import ButterworthErrorHighpass
            command._leg_error_highpass=ButterworthErrorHighpass(cutoff_hz,env.step_dt)
        command._leg_error_highpass_updates=0
        command._leg_error_highpass_trace=[]
        command._leg_error_highpass_settings=(cutoff_hz,env.step_dt,filter_order)
    if command._leg_error_highpass_settings != (cutoff_hz,env.step_dt,filter_order):
        raise ValueError('Highpass settings changed during environment lifetime')
    ids=command._highpass_leg_ids
    error=command.robot_joint_pos[:,ids]-command.joint_pos[:,ids]
    reset=env.episode_length_buf<=1
    high=command._leg_error_highpass.update(error,reset)
    command._leg_error_highpass_updates+=1
    root=command.cfg.body_names.index('base_link')
    speed=torch.linalg.vector_norm(command.body_lin_vel_w[:,root,:2],dim=-1)
    u=((speed-.05)/.05).clamp(0.,1.)
    gate=u*u*(3.-2.*u)
    cost=gate*(high/scale).square().mean(-1)
    if len(command._leg_error_highpass_trace)<trace_steps:
        values={'error':error,'high':high,'reset':reset,'gate':gate,'cost':cost}
        command._leg_error_highpass_trace.append({k:v.detach().cpu().numpy().copy() for k,v in values.items()})
    return cost
