"""Optional training action filter with independent episode state per environment."""
import torch

class VectorAnkleFilter:
    def __init__(self,names,num_envs,device,alpha=.5):
        if not 0<float(alpha)<=1:raise ValueError('alpha must be in (0,1]')
        self.mask=torch.tensor(['ankle' in n for n in names],device=device)
        if int(self.mask.sum())!=4:raise ValueError('Expected four ankle joints')
        self.alpha=float(alpha);self.valid=torch.zeros(num_envs,device=device,dtype=torch.bool);self.previous=None
        self.audit={'alpha':self.alpha,'scope':'ankles','selected_joints':[n for n in names if 'ankle' in n],'first_step':'identity','reset':'per_environment'}
    def __call__(self,actions):
        if actions.shape!=(len(self.valid),len(self.mask)):raise ValueError('Action dimensions changed')
        if self.previous is None:out=actions
        else:out=torch.where(self.valid[:,None]&self.mask[None,:],self.alpha*actions+(1.-self.alpha)*self.previous,actions)
        self.previous=out.clone();self.valid.fill_(True);return out
    def reset(self,dones):
        if dones.shape!=self.valid.shape:raise ValueError('Reset dimensions changed')
        self.valid[dones.bool()]=False

def wrap_environment(env,alpha):
    import gymnasium as gym
    class FilteredEnvironment(gym.Wrapper):
        def __init__(self,env):
            super().__init__(env)
            term=env.unwrapped.action_manager.get_term('JointPositionAction')
            self.action_filter=VectorAnkleFilter(list(term._joint_names),env.unwrapped.num_envs,env.unwrapped.device,alpha)
        def reset(self,**kwargs):
            self.action_filter.reset(torch.ones_like(self.action_filter.valid));return self.env.reset(**kwargs)
        def step(self,actions):
            result=self.env.step(self.action_filter(actions))
            self.action_filter.reset(result[2]|result[3]);return result
    return FilteredEnvironment(env)
