"""Issue19 observer: preserve issue18 metrics and record f05/f08 actual phase."""
import json
import numpy as np
import torch
from issue18_tracking import YawRecorder as Previous
class YawRecorder(Previous):
    def begin(self,label,target):
        super().begin(label,target)
        if self.condition == 'clean':
            self.full_series=True
        self.issue19_phase=[]
    def sample(self,*args,**kwargs):
        if self.label in ['forward_05','forward_08']:
            self.issue19_phase.append(self.projected_policy.pre_action_episode_step.detach().to(torch.int16).clone())
        return super().sample(*args,**kwargs)
    def finish(self,result):
        if self.issue19_phase:
            phase=torch.stack(self.issue19_phase).cpu().numpy()
            assert phase.shape==(1000,self.base.num_envs)
            np.savez_compressed(self.output/(self.label+'_phase.npz'),pre_action_episode_step=phase)
        return super().finish(result)
