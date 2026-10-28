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

def install(env, minimum_load=1., history_samples=1):
    """Sample after scene refresh, consume once before environment resets."""
    from unitree_rl_lab.tasks.mimic.mdp.point_slip import ground_point_slip,normalized_moving_point_slip
    validate_minimum_load(minimum_load)
    validate_history_samples(history_samples)
    base=env.unwrapped
    if hasattr(base,'_physics_slip_accumulator'): raise RuntimeError('Already installed')
    term=base.reward_manager.get_term_cfg('moving_point_slip')
    if term.func is not normalized_moving_point_slip: raise ValueError('Expected endpoint moving slip reward')
    accumulator=PhysicsSlipAccumulator(base.cfg.decimation)
    history=None
    if history_samples>1:
        from contact_history import ContactHistory
        from contact_feedback import contact_statistics
        history=ContactHistory(minimum_load,history_samples)
        original_reset=base._reset_idx
        def reset(ids):
            history.reset(ids)
            return original_reset(ids)
        base._reset_idx=reset
    def sample():
        if history is None:return ground_point_slip(base,minimum_load=minimum_load)
        robot=base.scene['robot'];data=robot.data;stats=[]
        for foot in ('left_foot','right_foot'):
            body=robot.body_names.index(foot)
            force,point,_,_,count,start=base.scene['slip_'+foot].contact_physx_view.get_contact_data(dt=base.cfg.sim.dt)
            stats.append(contact_statistics(force,point,count,start,data.body_com_pos_w[:,body],data.body_lin_vel_w[:,body],data.body_ang_vel_w[:,body]))
        load=torch.stack([v['load'] for v in stats],1)
        rms=torch.stack([v['xy_rms'] for v in stats],1)
        return history.update(load,rms,(load>1e-6).long()).sum(1)
    original_update=base.scene.update
    initial_counter=base._sim_step_counter
    accumulator.last_counter=initial_counter
    def update(*args,**kwargs):
        result=original_update(*args,**kwargs)
        counter=base._sim_step_counter
        if counter != accumulator.last_counter:
            accumulator.add(counter,sample())
        return result
    base._physics_slip_accumulator=accumulator
    base.scene.update=update
    term.func=physics_mean_moving_slip
    base.reward_manager.set_term_cfg('moving_point_slip',term)
    return {'original_function':'unitree_rl_lab.tasks.mimic.mdp.point_slip:normalized_moving_point_slip','active_function':'physics_slip_reward:physics_mean_moving_slip','sampling':'each physics step after scene.update','aggregation':'arithmetic mean over control period','decimation':base.cfg.decimation,'physics_dt':base.physics_dt,'weight':term.weight,'contact_history_samples':history_samples,'history_reset':'clear per environment before reset','velocity_source':'current positive contact points only','minimum_normal_load':float(minimum_load),'load_comparison':'strict_greater_than'}


def validate_minimum_load(value):
    if not math.isfinite(value) or value<=0:raise ValueError('Positive finite slip minimum load required')

def validate_load_resume(saved,requested,allow_change=False):
    validate_minimum_load(requested)
    previous=float(saved.get('minimum_normal_load',1.))
    validate_minimum_load(previous)
    if previous!=requested and not allow_change:raise ValueError('Resume slip load mismatch; explicit fork authorization flag required')
    return {'previous':previous,'requested':float(requested),'changed':previous!=requested,'explicit_fork':bool(allow_change)}


def validate_history_samples(value):
    if not isinstance(value,int) or isinstance(value,bool) or value<1:raise ValueError('Positive integer history samples required')

def validate_history_resume(saved,requested,allow_change=False):
    validate_history_samples(requested)
    previous=saved.get('contact_history_samples',1);validate_history_samples(previous)
    if previous!=requested and not allow_change:raise ValueError('Resume contact history mismatch; explicit fork authorization required')
    return {'previous':previous,'requested':requested,'changed':previous!=requested}
