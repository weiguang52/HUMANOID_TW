"""Translation-only camera follow: smooth horizontal travel, fixed height."""
import math
import numpy as np

class SmoothCameraFollow:
    def __init__(self, origin, dt, tau=.5):
        if not math.isfinite(tau) or tau < 0: raise ValueError('Camera tau must be nonnegative')
        self.origin=np.asarray(origin,dtype=float).copy()
        self.alpha=1.0 if tau==0 else -math.expm1(-dt/tau)

    def update(self, position, reset=False):
        position=np.asarray(position,dtype=float)
        if reset:
            self.origin[:]=position
        else:
            self.origin[:2]+=self.alpha*(position[:2]-self.origin[:2])
        return self.origin.copy()
