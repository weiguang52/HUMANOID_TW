"""Training-only paired observations and frozen-teacher relative action smoothness."""
import torch
PREVIOUS_POLICY = 'anchor_previous_policy'
PREVIOUS_VALID = 'anchor_previous_valid'

def temporal_anchor_mse(now, teacher_now, previous, teacher_previous, valid):
    if now.ndim != 2 or any(x.shape != now.shape for x in [teacher_now, previous, teacher_previous]):
        raise ValueError('Expected matching batch x action means')
    if valid.shape != (now.shape[0], 1):
        raise ValueError('Expected per-environment validity column')
    error = ((now-previous)-(teacher_now-teacher_previous)).square().mean(-1)
    mask = valid[:,0].to(error.dtype)
    return (error*mask).sum()/mask.sum().clamp_min(1)

class PreviousPolicyTraining:
    """Wrap an RSL VecEnv; auxiliary keys never enter actor/critic observation groups."""
    def __init__(self, env):
        self.env = env
        self.current_policy = None
        self.previous_policy = None
        self.valid = None
    def __getattr__(self, name):
        return getattr(self.env, name)
    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):
        self.env.episode_length_buf=value
    @torch.inference_mode()
    def _pack(self, obs, previous=None, valid=None):
        if PREVIOUS_POLICY in obs or PREVIOUS_VALID in obs:
            raise ValueError('Temporal observation wrapper already present')
        policy = obs['policy']
        if policy.ndim != 2 or policy.shape[0] != self.num_envs:
            raise ValueError('Expected flat per-environment policy observations')
        if previous is None:
            previous = policy
            valid = torch.zeros((self.num_envs,1),device=policy.device)
        if previous.shape != policy.shape or valid.shape != (self.num_envs,1):
            raise ValueError('Temporal observation dimensions changed')
        self.previous_policy = previous.detach().clone()
        self.valid = valid.detach().clone()
        self.current_policy = policy.detach().clone()
        out = obs.clone(False)
        out[PREVIOUS_POLICY] = self.previous_policy.clone()
        out[PREVIOUS_VALID] = self.valid.clone()
        return out
    def get_observations(self):
        obs = self.env.get_observations()
        return self._pack(obs,self.previous_policy,self.valid)
    def reset(self):
        obs,extras = self.env.reset()
        return self._pack(obs),extras
    @torch.inference_mode()
    def step(self, actions):
        if self.current_policy is None:
            self.get_observations()
        previous = self.current_policy.clone()
        obs,rewards,dones,extras = self.env.step(actions)
        if dones.shape != (self.num_envs,):
            raise ValueError('Unexpected done dimensions')
        valid = (~dones.bool()).float().unsqueeze(-1)
        return self._pack(obs,previous,valid),rewards,dones,extras
