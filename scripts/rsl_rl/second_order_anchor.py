"""Opt-in three-observation teacher-relative deterministic mean curvature.

Training-only history/loss, selected explicitly by temporal order2.
Auxiliary keys do not change actor or critic input groups.
"""
import torch
PREVIOUS='second_previous_policy'
OLDER='second_previous_previous_policy'
VALID='second_previous_valid'


def second_order_anchor_mse(now,teacher_now,previous,teacher_previous,older,teacher_older,valid):
    if now.ndim!=2 or any(x.shape!=now.shape for x in [teacher_now,previous,teacher_previous,older,teacher_older]):
        raise ValueError('Expected matching batch x action means')
    if valid.shape!=(now.shape[0],1):raise ValueError('Expected per-environment validity column')
    residual_now=now-teacher_now;residual_previous=previous-teacher_previous;residual_older=older-teacher_older
    error=(residual_now-2*residual_previous+residual_older).square().mean(-1)
    mask=valid[:,0].to(error.dtype)
    return (error*mask).sum()/mask.sum().clamp_min(1)


def student_mean_curvature_mse(now,previous,older,valid):
    """Direct deterministic mean curvature; independent of teacher curvature."""
    if now.ndim!=2 or previous.shape!=now.shape or older.shape!=now.shape:
        raise ValueError('Expected matching batch x action means')
    if valid.shape!=(now.shape[0],1):raise ValueError('Expected validity column')
    error=(now-2*previous+older).square().mean(-1)
    mask=valid[:,0].to(error.dtype)
    return (error*mask).sum()/mask.sum().clamp_min(1)


class PreviousTwoPoliciesTraining:
    """An RSL VecEnv proxy. Auxiliary keys are not actor or critic inputs.

    After each reset, action0/action1 are masked; action2 has two valid prior
    observations. Repeated get_observations does not advance history.
    """
    def __init__(self,env):
        self.env=env;self.current=None;self.previous=None;self.older=None;self.count=None
    def __getattr__(self,name):return getattr(self.env,name)
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):self.env.episode_length_buf=value
    @torch.inference_mode()
    def _pack(self,obs,previous=None,older=None,count=None):
        if any(k in obs for k in [PREVIOUS,OLDER,VALID]):raise ValueError('Second-order history already present')
        p=obs['policy']
        if p.ndim!=2 or p.shape[0]!=self.num_envs:raise ValueError('Expected flat per-environment policy inputs')
        if previous is None:
            previous=p;older=p;count=torch.zeros(self.num_envs,dtype=torch.long,device=p.device)
        if previous.shape!=p.shape or older.shape!=p.shape or count.shape!=(self.num_envs,):
            raise ValueError('Observation/history dimensions changed')
        self.current=p.detach().clone();self.previous=previous.detach().clone();self.older=older.detach().clone();self.count=count.detach().clone()
        out=obs.clone(False);out[PREVIOUS]=self.previous.clone();out[OLDER]=self.older.clone();out[VALID]=(self.count>=2).float().unsqueeze(-1)
        return out
    def get_observations(self):return self._pack(self.env.get_observations(),self.previous,self.older,self.count)
    def reset(self):
        obs,extras=self.env.reset();return self._pack(obs),extras
    @torch.inference_mode()
    def step(self,actions):
        if self.current is None:self.get_observations()
        previous=self.current.clone();older=self.previous.clone();count=self.count.clone()
        obs,rewards,dones,extras=self.env.step(actions)
        if dones.shape!=(self.num_envs,):raise ValueError('Unexpected done dimensions')
        count=torch.where(dones.bool(),0,(count+1).clamp_max(2))
        return self._pack(obs,previous,older,count),rewards,dones,extras
