"""Optional causal leg-action smoothing; state resets at episode boundaries."""
import torch
class LegActionFilter:
    def __init__(self,names,device,alpha=0.5,scope="legs"):
        if scope not in ("legs","ankles"):raise ValueError("Unknown filter scope")
        if not 0<float(alpha)<=1:raise ValueError('alpha must be in (0,1]')
        selected=[n.startswith(('left_','right_')) and any(t in n for t in ('hip','knee','ankle','foot_roll')) for n in names]
        if scope=='ankles':selected=['ankle' in n for n in names]
        expected=14 if scope=='legs' else 4
        if sum(selected)!=expected:raise ValueError('Unexpected filter joint count')
        self.mask=torch.tensor(selected,device=device);self.alpha=float(alpha);self.previous=None;self.previous_reference=None
        self.audit=dict(alpha=self.alpha,scope=scope,selected_joints=[n for n,s in zip(names,selected) if s],first_step='identity',reset='clear')
    def __call__(self,action,reference=None):
        if reference is not None:
            if reference.shape != action.shape:raise ValueError('Reference/action shape mismatch')
            if self.previous is None:
                self.previous=action.clone()
                self.previous_reference=reference.clone()
                return action
            predicted=self.previous+(reference-self.previous_reference)
            filtered=self.alpha*action+(1-self.alpha)*predicted
            out=torch.where(self.mask,filtered,action)
            self.previous=out.clone()
            self.previous_reference=reference.clone()
            return out
        if self.previous is None:self.previous=action.clone();return action
        out=torch.where(self.mask,self.alpha*action+(1-self.alpha)*self.previous,action)
        self.previous=out.clone();return out
    def reset(self,dones):
        if dones.any():self.previous=None;self.previous_reference=None

class TwoTapAnkleFilter(LegActionFilter):
    """Two raw action samples; no recursively filtered history."""
    def __call__(self,action,reference=None):
        if reference is not None:raise ValueError('Two-tap filter does not use reference input')
        if self.previous is None:
            self.previous=action.clone()
            return action
        out=torch.where(self.mask,self.alpha*action+(1-self.alpha)*self.previous,action)
        self.previous=action.clone()
        return out
