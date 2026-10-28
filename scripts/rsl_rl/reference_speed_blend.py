"""Explicit two-policy experiment gated only by reference planar speed."""
import math,torch,numpy as np
from policy_action_blend import PolicyActionBlend

def candidate_weight(speed,maximum=.5,low=.05,high=.1):
    if not 0<=maximum<=1 or not 0<=low<high:raise ValueError('Invalid gate parameters')
    u=((speed-low)/(high-low)).clamp(0,1)
    return maximum*u*u*(3-2*u)

def mix_actions(teacher,candidate,weight):
    if teacher.shape!=candidate.shape:raise ValueError('Action shapes differ')
    if not torch.isfinite(weight).all():raise ValueError('Nonfinite gate')
    if bool((weight==0).all()):return teacher
    if bool((weight==1).all()):return candidate
    return teacher+(candidate-teacher)*weight[...,None]

class ReferenceSpeedBlend:
    def __init__(self,path,device,command,maximum=.5,low=.05,high=.1,hip_alpha=None,action_names=None,hip_mode='absolute',leg_alpha=None,leg_mode='absolute',candidate_scope='all'):
        if not all(math.isfinite(x) for x in [maximum,low,high]):raise ValueError('Nonfinite parameters')
        candidate_weight(torch.zeros(1),maximum,low,high)
        self.base=PolicyActionBlend(path,1.,device)
        self.command=command;self.index=command.cfg.body_names.index('base_link')
        self.maximum=maximum;self.low=low;self.high=high;self.trace=[]
        self.hip_filter=None;self.hip_trace=[]
        self.leg_filter=None;self.leg_trace=[];self.leg_mode=leg_mode;self.leg_reference_trace=[]
        if leg_mode not in ['absolute','teacher_residual']:raise ValueError('Unknown candidate leg filter mode')
        if leg_alpha is None and leg_mode!='absolute':raise ValueError('Leg mode needs leg filter alpha')
        if hip_alpha is not None and leg_alpha is not None:raise ValueError('Do not combine candidate hip and leg filters')
        if leg_alpha is not None:
            from candidate_leg_filter import CandidateLegFilter
            if action_names is None:raise ValueError('Leg filter needs action joint order')
            if leg_mode=='absolute':self.leg_filter=CandidateLegFilter(action_names,device,float(leg_alpha))
            else:
                from candidate_leg_filter import TeacherResidualLegFilter
                self.leg_filter=TeacherResidualLegFilter(action_names,device,float(leg_alpha))
        self.hip_mode=hip_mode;self.hip_reference_trace=[]
        if hip_alpha is not None:
            from candidate_hip_filter import CandidateHipFilter
            if action_names is None:raise ValueError('Hip filter needs action joint order')
            if hip_mode=='absolute':self.hip_filter=CandidateHipFilter(action_names,device,float(hip_alpha))
            elif hip_mode=='teacher_residual':
                from candidate_hip_filter import TeacherResidualHipFilter
                self.hip_filter=TeacherResidualHipFilter(action_names,device,float(hip_alpha))
            else:raise ValueError('Unknown candidate hip filter mode')
        self.action_scope=None
        if candidate_scope!='all':
            if hip_alpha is not None or leg_alpha is not None:raise ValueError('Action partition cannot combine candidate filters')
            from candidate_action_scope import make_scope_contract
            self.action_scope=make_scope_contract(action_names,candidate_scope)
        self.audit=dict(self.base.audit)
        self.audit.update(mode='reference_speed',maximum_candidate_weight=maximum,speed_low_m_s=low,speed_high_m_s=high,rule='smoothstep reference base planar speed; no clip labels',teacher_weight='1-candidate_weight')
        if self.action_scope is not None:self.audit['candidate_action_scope']=self.action_scope
        if self.hip_filter is not None:self.audit['candidate_hip_filter']=self.hip_filter.audit
        if self.leg_filter is not None:self.audit['candidate_leg_filter']=self.leg_filter.audit
    @torch.inference_mode()
    def __call__(self,obs,action):
        speed=torch.linalg.vector_norm(self.command.body_lin_vel_w[:,self.index,:2],dim=-1)
        weight=candidate_weight(speed,self.maximum,self.low,self.high)
        self.audit['observation_key']='contact_base_policy' if 'contact_base_policy' in obs else 'policy'
        teacher=self.base.teacher(obs.get('contact_base_policy',obs['policy']))
        if self.hip_filter is not None:
            raw=action
            if self.hip_mode=='teacher_residual':
                action=self.hip_filter(action,teacher)
                self.hip_reference_trace.append(teacher[:,self.hip_filter.mask].cpu().numpy())
            else:action=self.hip_filter(action)
            self.hip_trace.append(torch.stack([raw[:,self.hip_filter.mask],action[:,self.hip_filter.mask]],dim=-1).cpu().numpy())
        if self.leg_filter is not None:
            raw=action
            if self.leg_mode=='teacher_residual':
                action=self.leg_filter(action,teacher)
                self.leg_reference_trace.append(teacher[:,self.leg_filter.mask].cpu().numpy())
            else:action=self.leg_filter(action)
            self.leg_trace.append(torch.stack([raw[:,self.leg_filter.mask],action[:,self.leg_filter.mask]],dim=-1).cpu().numpy())
        self.trace.append(torch.stack([speed,weight],dim=-1).cpu().numpy())
        scope=getattr(self,'action_scope',None)
        if scope is None:return mix_actions(teacher,action,weight)
        from candidate_action_scope import mix_scoped_actions
        return mix_scoped_actions(teacher,action,weight,scope['scope'],scope['action_joint_names'])
    def reset(self,dones):
        if self.hip_filter is not None:self.hip_filter.reset(dones)
        if self.leg_filter is not None:self.leg_filter.reset(dones)
    def save_trace(self,path):
        data={'speed_and_candidate_weight':np.asarray(self.trace)}
        if self.hip_filter is not None:data['candidate_hip_raw_filtered']=np.asarray(self.hip_trace)
        if self.hip_filter is not None and self.hip_mode=='teacher_residual':data['candidate_hip_reference']=np.asarray(self.hip_reference_trace)
        if self.leg_filter is not None:
            data['candidate_leg_raw_filtered']=np.asarray(self.leg_trace)
            data['candidate_leg_joints']=np.asarray(self.leg_filter.audit['joints'])
            if self.leg_mode=='teacher_residual':data['candidate_leg_reference']=np.asarray(self.leg_reference_trace)
        np.savez_compressed(path,**data)
