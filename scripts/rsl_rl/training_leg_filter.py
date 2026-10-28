"""Opt-in training leg filter before the existing ankle filter."""
import math
from candidate_leg_filter import CandidateLegFilter

def validate_resume_contract(saved,alpha):
    if saved is None:return
    if saved.get('mode')!='absolute' or saved.get('scope')!='legs14':
        raise ValueError('Unsupported saved training leg filter contract')
    if alpha is None or not math.isfinite(alpha) or alpha!=saved['alpha']:
        raise ValueError('Resume must preserve checkpoint training leg filter')
    if saved.get('order')!='candidate legs then existing ankles':
        raise ValueError('Unsupported saved training filter order')

def wrap_leg_environment(env,alpha):
    import gymnasium as gym
    import torch
    class LegFilteredEnvironment(gym.Wrapper):
        def __init__(self,env):
            super().__init__(env)
            base=env.unwrapped
            term=base.action_manager.get_term('JointPositionAction')
            self.trace=[]
            self.leg_action_filter=CandidateLegFilter(list(term._joint_names),base.device,alpha)
            self.leg_action_filter.audit.update(mode='absolute',order='candidate legs then existing ankles',branch='training before existing ankle filter')
        def reset(self,**kwargs):
            base=self.unwrapped
            self.leg_action_filter.reset(torch.ones(base.num_envs,device=base.device,dtype=torch.bool))
            return self.env.reset(**kwargs)
        def step(self,actions):
            capture=len(self.trace)<64
            leg_valid=(torch.zeros(actions.shape[0],device=actions.device,dtype=torch.bool)
                       if self.leg_action_filter.valid is None else self.leg_action_filter.valid.clone())
            ankle_valid=self.env.action_filter.valid.clone()
            filtered=self.leg_action_filter(actions)
            result=self.env.step(filtered)
            done=result[2]|result[3]
            if capture:
                values={'raw':actions,'leg':filtered,'ankle':self.env.action_filter.previous,
                        'leg_valid':leg_valid,'ankle_valid':ankle_valid,'done':done}
                self.trace.append({k:v.detach().cpu().numpy().copy() for k,v in values.items()})
            self.leg_action_filter.reset(done)
            return result
        def save_trace(self,path):
            import numpy as np
            if not self.trace:raise ValueError('No training leg-filter trace collected')
            data={k:np.stack([row[k] for row in self.trace]) for k in self.trace[0]}
            data['joint_names']=np.asarray(self.unwrapped.action_manager.get_term('JointPositionAction')._joint_names)
            np.savez_compressed(path,**data)
    return LegFilteredEnvironment(env)
