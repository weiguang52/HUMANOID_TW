"""Opt-in physics-rate slip sampling; install only for an explicit experiment."""
import math
import torch

class PhysicsSlipAccumulator:
    def __init__(self, decimation):
        if decimation < 1: raise ValueError('Positive decimation required')
        self.decimation=decimation
        self.last_counter=None
        self.count=0
        self.total=None
        self.completed_windows=0
    def add(self, counter, value):
        if counter == self.last_counter: return
        if self.last_counter is not None and counter != self.last_counter+1:
            raise RuntimeError('Skipped physics sample')
        if self.count >= self.decimation: raise RuntimeError('Unconsumed reward window')
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite slip sample')
        self.total=value.clone() if self.count==0 else self.total+value
        self.count+=1
        self.last_counter=counter
    def consume(self):
        if self.count != self.decimation: raise RuntimeError('Incomplete physics reward window')
        result=self.total/self.decimation
        self.count=0
        self.total=None
        self.completed_windows+=1
        return result

def physics_mean_moving_slip(env,command_name='motion',speed_scale=.05):
    from unitree_rl_lab.tasks.mimic.mdp.point_slip import moving_slip_gate
    if speed_scale<=0: raise ValueError('Positive speed normalization required')
    command=env.command_manager.get_term(command_name)
    index=command.cfg.body_names.index('base_link')
    speed=torch.linalg.vector_norm(command.body_lin_vel_w[:,index,:2],dim=-1)
    return moving_slip_gate(speed)*env._physics_slip_accumulator.consume()/(2*speed_scale)

def install(env, minimum_load=1.):
    """Sample after scene refresh, consume once before environment resets."""
    from unitree_rl_lab.tasks.mimic.mdp.point_slip import ground_point_slip,normalized_moving_point_slip
    validate_minimum_load(minimum_load)
    base=env.unwrapped
    if hasattr(base,'_physics_slip_accumulator'): raise RuntimeError('Already installed')
    term=base.reward_manager.get_term_cfg('moving_point_slip')
    if term.func is not normalized_moving_point_slip: raise ValueError('Expected endpoint moving slip reward')
    accumulator=PhysicsSlipAccumulator(base.cfg.decimation)
    original_update=base.scene.update
    initial_counter=base._sim_step_counter
    accumulator.last_counter=initial_counter
    def update(*args,**kwargs):
        result=original_update(*args,**kwargs)
        counter=base._sim_step_counter
        if counter != accumulator.last_counter:
            accumulator.add(counter,ground_point_slip(base,minimum_load=minimum_load))
        return result
    base._physics_slip_accumulator=accumulator
    base.scene.update=update
    term.func=physics_mean_moving_slip
    base.reward_manager.set_term_cfg('moving_point_slip',term)
    return {'original_function':'unitree_rl_lab.tasks.mimic.mdp.point_slip:normalized_moving_point_slip','active_function':'physics_slip_reward:physics_mean_moving_slip','sampling':'each physics step after scene.update','aggregation':'arithmetic mean over control period','decimation':base.cfg.decimation,'physics_dt':base.physics_dt,'weight':term.weight,'minimum_normal_load':float(minimum_load),'load_comparison':'strict_greater_than'}


def validate_minimum_load(value):
    if not math.isfinite(value) or value<=0:raise ValueError('Positive finite slip minimum load required')

def validate_load_resume(saved,requested,allow_change=False):
    validate_minimum_load(requested)
    previous=float(saved.get('minimum_normal_load',1.))
    validate_minimum_load(previous)
    if previous!=requested and not allow_change:raise ValueError('Resume slip load mismatch; explicit fork authorization flag required')
    return {'previous':previous,'requested':float(requested),'changed':previous!=requested,'explicit_fork':bool(allow_change)}
