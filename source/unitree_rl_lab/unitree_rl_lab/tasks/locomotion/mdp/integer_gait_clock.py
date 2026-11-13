"""Opt-in TW20-step gait clock shared by rewards and observations."""
import math
import torch

def integer_clock_enabled(env):
    return bool(getattr(getattr(env,'cfg',None),'issue17_integer_clock',False))

def period_steps(env,period):
    count=round(period/env.step_dt)
    assert count==20 and math.isclose(period/env.step_dt,count,rel_tol=0,abs_tol=1e-8)
    return count

def integer_gait_phase(env,period):
    count=period_steps(env,period)
    return (env.episode_length_buf%count).to(torch.float32)/count

def integer_leg_stance(env,period,offset,threshold):
    count=period_steps(env,period);boundary=round(threshold*count)
    assert boundary==11 and math.isclose(threshold*count,boundary,rel_tol=0,abs_tol=1e-8)
    shifts=[round(v*count) for v in offset]
    assert shifts==[0,10] and all(math.isclose(v*count,k,rel_tol=0,abs_tol=1e-8) for v,k in zip(offset,shifts))
    idx=env.episode_length_buf%count
    leg=(idx[:,None]+torch.tensor(shifts,device=idx.device,dtype=idx.dtype))%count
    return leg<boundary
