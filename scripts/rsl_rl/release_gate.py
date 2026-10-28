"""Optional per-environment release-only reference gate state."""
import math
import torch
class TorchReleaseGate:
    def __init__(self,tau=.2,dt=.02):
        if not math.isfinite(tau) or tau<0 or not math.isfinite(dt) or dt<=0:
            raise ValueError('Nonnegative finite tau and positive dt required')
        self.tau=float(tau);self.decay=math.exp(-dt/tau) if tau else 0.;self.previous=None
    @torch.inference_mode()
    def step(self,raw):
        if raw.ndim!=1 or not torch.isfinite(raw).all() or (raw<0).any() or (raw>1).any():
            raise ValueError('Expected bounded per-environment weights')
        if self.tau==0:return raw
        if self.previous is not None and (raw.shape!=self.previous.shape or raw.device!=self.previous.device or raw.dtype!=self.previous.dtype):
            raise ValueError('Gate input contract changed')
        result=raw.clone() if self.previous is None else torch.maximum(raw,self.previous*self.decay)
        self.previous=result.clone();return result
    @torch.inference_mode()
    def reset(self,dones):
        if self.previous is None:return
        if dones.shape!=self.previous.shape:raise ValueError('Unexpected reset shape')
        self.previous.masked_fill_(dones.bool(),0.)
