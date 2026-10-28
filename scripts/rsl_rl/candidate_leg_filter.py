"""Optional candidate-only fourteen-leg-action IIR, before speed gating."""
import math
import torch

class CandidateLegFilter:
    def __init__(self,names,device,alpha):
        if not math.isfinite(alpha) or not 0<alpha<=1:raise ValueError('Invalid leg filter alpha')
        expected=[side+'_'+joint for side in ['left','right'] for joint in ['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_pitch_joint','ankle_yaw_joint','ankle_pitch_joint','foot_roll']]
        if len(set(names))!=len(names) or any(names.count(n)!=1 for n in expected):
            raise ValueError('Fourteen unique expected leg actions required')
        self.mask=torch.tensor([n in expected for n in names],device=device);self.alpha=float(alpha);self.previous=None;self.valid=None
        self.audit={'alpha':self.alpha,'joints':[n for n in names if n in expected],'scope':'legs14','branch':'candidate before speed blend','first_step':'identity','reset':'per environment'}
    @torch.inference_mode()
    def __call__(self,action):
        if action.ndim!=2 or action.shape[1]!=len(self.mask):raise ValueError('Unexpected action shape')
        if self.previous is None:
            self.previous=action.clone();self.valid=torch.ones(action.shape[0],device=action.device,dtype=torch.bool);return action
        if self.previous.shape!=action.shape:raise ValueError('Environment shape changed')
        filtered=self.alpha*action+(1-self.alpha)*self.previous
        out=torch.where(self.mask[None,:]&self.valid[:,None],filtered,action)
        self.previous=out.clone();self.valid.fill_(True);return out
    @torch.inference_mode()
    def reset(self,dones):
        if self.valid is not None:
            if dones.shape!=self.valid.shape:raise ValueError('Unexpected reset shape')
            self.valid[dones.bool()]=False


class TeacherResidualLegFilter(CandidateLegFilter):
    """Smooth only candidate-minus-teacher; preserve the current teacher action."""
    def __init__(self,names,device,alpha):
        super().__init__(names,device,alpha)
        self.audit.update(mode='teacher_residual',reference='current original teacher action')
    @torch.inference_mode()
    def __call__(self,action,teacher):
        if action.shape!=teacher.shape:raise ValueError('Teacher/action shape differs')
        valid=None if self.valid is None else self.valid.clone()
        filtered=super().__call__(action-teacher)
        if valid is None or self.alpha==1.:return action
        return torch.where(self.mask[None,:]&valid[:,None],teacher+filtered,action)
