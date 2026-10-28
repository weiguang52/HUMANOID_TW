"""Opt-in candidate-branch hip pitch/roll filter; teacher branch is separate."""
import math,torch

class CandidateHipFilter:
    def __init__(self,names,device,alpha):
        if not math.isfinite(alpha) or not 0<alpha<=1:raise ValueError('Invalid hip filter alpha')
        expected=[s+'_'+j+'_joint' for s in ['left','right'] for j in ['hip_pitch','hip_roll']]
        if any(names.count(n)!=1 for n in expected):raise ValueError('Four unique hip joints required')
        self.mask=torch.tensor([n in expected for n in names],device=device)
        self.alpha=float(alpha);self.previous=None;self.valid=None
        self.audit={'alpha':self.alpha,'joints':[n for n in names if n in expected],'branch':'candidate before speed blend','first_step':'identity','reset':'per environment'}
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

class TeacherResidualHipFilter:
    def __init__(self,names,device,alpha):
        self.filter=CandidateHipFilter(names,device,alpha)
        self.mask=self.filter.mask
        self.audit=dict(self.filter.audit,mode='teacher_residual',reference='current frozen teacher action')
    def __call__(self,action,teacher):
        if action.shape!=teacher.shape:raise ValueError('Teacher/action shape mismatch')
        valid=self.filter.valid.clone() if self.filter.valid is not None else torch.zeros(action.shape[0],device=action.device,dtype=torch.bool)
        residual=self.filter(action-teacher)
        return torch.where(self.mask[None,:]&valid[:,None],teacher+residual,action)
    def reset(self,dones):
        self.filter.reset(dones)
