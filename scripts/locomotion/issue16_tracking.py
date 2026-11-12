"""Issue16 heading observer, reusing unchanged issue15 metrics and physical/RNG checks."""
import torch
from issue15_tracking import YawRecorder as PreviousYawRecorder

class YawRecorder(PreviousYawRecorder):
    def __init__(self,*args,mode='mirror',**kwargs):
        self.mode=mode
        self.default_target_error_max=0.
        super().__init__(*args,**kwargs)
    def begin(self,label,target):
        super().begin(label,target)
        # Stand is a primary heading-gate command; keep every quaternion-yaw sample too.
        if label=='stand':self.full_series=True
    def sample(self,*args,**kwargs):
        if self.mode=='default':
            term=self.base.action_manager.get_term('JointPositionAction')
            target=term.processed_actions
            expected=self.robot.data.default_joint_pos[:,term._joint_ids]
            error=float((target-expected).abs().max())
            self.default_target_error_max=max(self.default_target_error_max,error)
            assert error<1e-7,('Default PD target changed',error)
        return super().sample(*args,**kwargs)
    def job_metadata(self):
        result=super().job_metadata()
        result.update(version='issue16_heading_reflection_v1',mode=self.mode,default_target_error_max=self.default_target_error_max,
                      original_issue15_heading_and_metric_definitions_preserved=True)
        return result
