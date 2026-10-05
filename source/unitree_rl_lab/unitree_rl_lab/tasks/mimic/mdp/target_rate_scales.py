"""Explicit per-joint target-rate factors; all factors remain within actuator caps."""
import math


def resolve_scales(names, default, overrides=None):
    overrides=overrides or {}
    if set(overrides)-set(names):raise ValueError('Unknown target-rate joint')
    values=[overrides.get(n,default) for n in names]
    if any(not math.isfinite(v) or not 0<v<=1 for v in values):
        raise ValueError('Target-rate factors must be in (0,1]')
    return values
