"""Opt-in fourth-order causal tracking-error reward filter."""
import math
import torch

class ButterworthErrorHighpass:
    def __init__(self,cutoff_hz,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Positive finite dt required')
        if not math.isfinite(cutoff_hz) or not 0<cutoff_hz<.5/dt:
            raise ValueError('Cutoff must lie between zero and Nyquist')
        from scipy.signal import butter,sosfilt_zi
        self.sos=butter(4,cutoff_hz,btype='highpass',fs=1./dt,output='sos')
        self.zi=sosfilt_zi(self.sos)
        self.state=None

    def update(self,error,reset):
        if error.ndim!=2 or error.dtype not in (torch.float32,torch.float64):
            raise ValueError('Expected environment by joint floating error')
        if reset.shape!=error.shape[:1] or reset.dtype!=torch.bool or reset.device!=error.device:
            raise ValueError('Expected matching boolean per-environment reset')
        x=error.detach()
        if self.state is None:
            self.coefficients=torch.as_tensor(self.sos,dtype=x.dtype,device=x.device)
            self.initial=torch.as_tensor(self.zi,dtype=x.dtype,device=x.device)
            self.state=self.initial[:,:,None,None]*x[None,None]
            reset=torch.ones_like(reset)
        if self.state.shape[2:]!=x.shape or self.state.dtype!=x.dtype or self.state.device!=x.device:
            raise ValueError('Environment, joint, dtype or device changed')
        seeded=self.initial[:,:,None,None]*x[None,None]
        self.state=torch.where(reset[None,None,:,None],seeded,self.state)
        states=[]
        for i in range(len(self.coefficients)):
            b0,b1,b2,a0,a1,a2=self.coefficients[i].unbind()
            y=b0*x+self.state[i,0]
            z1=b1*x-a1*y+self.state[i,1]
            z2=b2*x-a2*y
            states.append(torch.stack([z1,z2]));x=y
        self.state=torch.stack(states).detach()
        return torch.where(reset[:,None],torch.zeros_like(x),x)
