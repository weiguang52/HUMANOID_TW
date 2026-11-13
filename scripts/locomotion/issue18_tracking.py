"""Observational checks of actual action feedback and the pre-action episode clock."""
import json
import numpy as np
import torch
from issue16_tracking import YawRecorder as PreviousYawRecorder
class YawRecorder(PreviousYawRecorder):
    def begin(self,label,target):
        super().begin(label,target)
        assert torch.equal(self.base.episode_length_buf,torch.zeros_like(self.base.episode_length_buf)), 'Reset episode clock is not zero'
        self.reconstructed_step=torch.zeros_like(self.base.episode_length_buf)
        self.episode_steps=[]
        self.applied_checks=0
        self.applied_max_error=0.0
    def sample(self,values,valid,velocity,wz):
        policy=self.projected_policy
        assert torch.equal(policy.pre_action_episode_step,self.reconstructed_step), 'Integer phase reconstruction mismatch'
        if bool(valid.any()):
            actual=self.base.action_manager.action[valid]
            expected=policy.last_output[valid]
            error=float((actual-expected).abs().max());self.applied_max_error=max(self.applied_max_error,error)
            assert torch.equal(actual,expected), 'Wrapper output differs from actually applied raw action (clipping or processing)'
        self.applied_checks+=1
        if self.label=='forward_05':self.episode_steps.append(policy.pre_action_episode_step.detach().to(torch.int16).clone())
        # The unchanged evaluator defines valid exactly as ~dones. A terminal
        # transition resets the next policy's episode clock and is not a yaw sample.
        self.reconstructed_step=torch.where(valid,self.reconstructed_step+1,torch.zeros_like(self.reconstructed_step))
        return super().sample(values,valid,velocity,wz)
    def finish(self,result):
        if self.label=='forward_05':
            steps=torch.stack(self.episode_steps).cpu().numpy()
            assert steps.shape==(1000,self.base.num_envs)
            np.savez_compressed(self.output/(self.label+'_phase.npz'),pre_action_episode_step=steps)
        (self.output/(self.label+'_feedback.json')).write_text(json.dumps(dict(passed=True,actual_last_action_observation_checks=self.projected_policy.feedback_checks,last_action_max_error=self.projected_policy.feedback_max_error,actually_applied_action_checks=self.applied_checks,applied_action_max_error=self.applied_max_error,pre_action_episode_step_reconstruction_exact=True,clock_reset_zero=True,phase_bin_convention='pre-action episode_length_buf %20, left0..9 right10..19; yaw is post-step increment; terminal steps excluded'),indent=2)+'\n')
        return super().finish(result)
    def job_metadata(self):
        result=super().job_metadata();result.update(issue18_actual_feedback_verified=True,issue18_pre_action_phase_reconstruction_verified=True,version='issue18_strict_projection_observer_v1');return result
