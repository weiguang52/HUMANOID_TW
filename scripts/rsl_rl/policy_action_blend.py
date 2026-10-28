"""Explicit frozen two-policy action interpolation for controlled evaluation."""
import hashlib,math
from pathlib import Path
import torch

class PolicyActionBlend:
    def __init__(self,path,weight,device,joint_names=None,scope="all"):
        if scope not in ("all","arms"):raise ValueError("Unknown blend scope")
        weight=float(weight)
        if not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError('Teacher weight must be finite and in [0,1]')
        p=Path(path);self.weight=weight
        self.mask=None
        if scope=="arms":
            if joint_names is None:raise ValueError("Action joint names required")
            selected=[any(t in n for t in ("shoulder","elbow","wrist")) for n in joint_names]
            if sum(selected)!=10:raise ValueError(f"Expected 10 arm actions, got {sum(selected)}")
            self.mask=torch.tensor(selected,device=device,dtype=torch.bool)
        self.teacher=torch.jit.load(str(p),map_location=device).eval()
        self.audit={'teacher_export':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'teacher_weight':weight,'observation_key':'policy'}
        self.audit.update(scope=scope,action_joint_names=joint_names,selected_joints=None if self.mask is None else [n for n,m in zip(joint_names,selected) if m])
    @torch.inference_mode()
    def __call__(self,obs,action):
        if self.weight==0:return action
        self.audit['observation_key']='contact_base_policy' if 'contact_base_policy' in obs else 'policy'
        teacher=self.teacher(obs.get('contact_base_policy',obs['policy']))
        if teacher.shape!=action.shape:raise ValueError('Teacher action shape mismatch')
        mixed=teacher if self.weight==1 else action*(1-self.weight)+teacher*self.weight
        if self.mask is None:return mixed
        if action.shape[-1]!=len(self.mask):raise ValueError("Action order dimension mismatch")
        return torch.where(self.mask,mixed,action)
