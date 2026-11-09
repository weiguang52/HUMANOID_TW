"""Issue #7 observational hooks. No sampling, physics or policy changes."""
import json,math
from pathlib import Path
import numpy as np
import torch
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

def recorded_push(env, env_ids, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    robot=env.scene[asset_cfg.name]
    before=robot.data.root_vel_w[env_ids].clone()
    observer=getattr(env,"_issue7_recorder",None)
    if observer is not None and hasattr(observer,"before_push"):
        observer.before_push(env_ids)
    push_by_setting_velocity(env,env_ids,velocity_range,asset_cfg)
    recorder=getattr(env,"_issue7_recorder",None)
    if recorder is not None:
        delta=robot.data.root_vel_w[env_ids,:3]-before[:,:3]
        recorder.push(env_ids,delta,robot.data.root_quat_w[env_ids])

from confidence_stats import wilson

class FailureRecorder:
    def __init__(self,base,output):
        self.env=base;self.robot=base.scene["robot"];self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.dt=base.step_dt;self.length=round(1/self.dt)+1
        self.feet=[base.scene["contact_forces"].body_names.index(x) for x in ["left_foot","right_foot"]]
        self.foot_bodies=[self.robot.body_names.index(x) for x in ["left_foot","right_foot"]]
        self.action=base.action_manager.get_term("JointPositionAction")
        self.original=base.termination_manager.compute
        self.recording=False
        base.termination_manager.compute=self.compute
        base._issue7_recorder=self
    def begin(self,label,target):
        self.label=label;self.target=list(target);self.step=0
        n=self.env.num_envs;device=self.env.device
        self.failed=torch.zeros(n,dtype=torch.bool,device=device)
        self.last_push_step=torch.full((n,),-100000,dtype=torch.long,device=device)
        self.last_push_w=torch.zeros((n,3),device=device)
        self.last_push_b=torch.zeros((n,3),device=device)
        self.last_push_contact=torch.zeros((n,2),device=device)
        self.previous_contact=torch.zeros((n,2),dtype=torch.bool,device=device)
        self.onset=torch.full((n,2),-100000,dtype=torch.long,device=device)
        self.period=torch.zeros((n,2),device=device)
        self.rows=[];self.traces=[];self.records=[];self.exposure={}
        self.initial_delay=self.action.delay[:,0].clone()
        self.ring=None;self.recording=True
    def push(self,ids,delta,q):
        if not self.recording:return
        self.last_push_step[ids]=self.step
        self.last_push_w[ids]=delta
        self.last_push_b[ids]=quat_apply_inverse(q,delta)
        self.last_push_contact[ids]=self.previous_contact[ids].float()
    def compute(self):
        done=self.original()
        if not self.recording:return done
        self.step+=1
        r=self.robot;env=self.env
        contact=env.scene["contact_forces"].data.net_forces_w[:,self.feet,:].norm(dim=-1)>1.
        rising=contact&~self.previous_contact
        period=self.step-self.onset
        self.period=torch.where(rising&(self.onset>0),period.float(),self.period)
        self.onset=torch.where(rising,self.step,self.onset)
        phase=torch.where(self.period>0,((self.step-self.onset)/self.period.clamp_min(1)).clamp(0,1),-1.)
        self.previous_contact.copy_(contact)
        q=r.data.root_quat_w
        roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        pitch=torch.asin((2*(q[:,0]*q[:,2]-q[:,3]*q[:,1])).clamp(-1,1))
        limits=r.data.soft_joint_pos_limits
        near=(r.data.joint_pos<=limits[:,:,0]+.02)|(r.data.joint_pos>=limits[:,:,1]-.02)
        hard=r.data.joint_pos_limits
        hard_near=(r.data.joint_pos<=hard[:,:,0]+.02)|(r.data.joint_pos>=hard[:,:,1]-.02)
        age=torch.where(self.last_push_step>0,(self.step-self.last_push_step)*self.dt,-1.)
        delayed=self.action.delay[:,0].float()
        height=r.data.root_pos_w[:,2]-env.scene.env_origins[:,2]
        pose=torch.cat((r.data.root_pos_w-env.scene.env_origins,q),-1)
        values=torch.cat((pose,r.data.joint_pos,r.data.root_vel_w,
            height[:,None],roll[:,None],pitch[:,None],contact.float(),phase,
            env.scene["contact_forces"].data.net_forces_w[:,self.feet,:].norm(dim=-1),
            near.float(),hard_near.float(),delayed[:,None],self.last_push_w,self.last_push_b,age[:,None],
            self.last_push_contact,torch.tensor(self.target,device=env.device).expand(env.num_envs,3)),dim=-1)
        if self.ring is None:
            self.ring=torch.zeros((self.length,env.num_envs,values.shape[-1]),device=env.device)
            j=len(r.joint_names);k=0;self.fields={}
            for name,width in [('root_pose_local',7),('joint_pos',j),('root_velocity_world',6),('base_height',1),('roll',1),('pitch',1),('contact',2),('cycle_phase',2),('foot_force_norm',2),('joint_near_soft_limit',j),('joint_near_hard_limit',j),('delay20ms',1),('last_push_delta_world',3),('last_push_delta_body',3),('seconds_since_push',1),('contact_at_push',2),('command',3)]:
                self.fields[name]=[k,k+width];k+=width
            assert k==values.shape[-1]
        self.ring[(self.step-1)%self.length]=values
        fresh=env.termination_manager.terminated&~self.failed
        # Exposure is at-risk pre-reset state, including the first event step.
        at_risk=~self.failed
        phasecode=contact[:,0].long()+2*contact[:,1].long()
        for name,bins in [('delay',delayed.long()),('contact_phase',phasecode),('near_limit',near.any(-1).long()),('recent_push',((age>=0)&(age<=1)).long())]:
            counts=torch.bincount(bins[at_risk],minlength=4).cpu().tolist()
            if name not in self.exposure:self.exposure[name]=[0]*len(counts)
            self.exposure[name]=[a+b for a,b in zip(self.exposure[name],counts)]
        ids=fresh.nonzero().flatten()
        if ids.numel():
            count=min(self.step,self.length)
            order=(torch.arange(self.step-count,self.step,device=env.device)%self.length).long()
            traces=self.ring[order][:,ids].transpose(0,1).cpu().numpy()
            for idx,trace in zip(ids.cpu().tolist(),traces):
                terms=[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[idx])]
                record={'env_id':idx,'command':self.label,'target':self.target,'failure_step':self.step,'failure_second':self.step*self.dt,
                    'termination_terms':terms,'delay_ms':20*int(delayed[idx].item()),'trace_frames':count,
                    'trace_relative_seconds':[(x-count+1)*self.dt for x in range(count)]}
                self.records.append(record);self.traces.append(trace)
        self.failed|=fresh
        return done
    def finish(self):
        self.recording=False
        # Fixed-length archive with validity mask for early failures.
        width=self.ring.shape[-1]
        traces=np.zeros((len(self.traces),self.length,width),dtype=np.float32)
        valid=np.zeros((len(self.traces),self.length),dtype=bool)
        for i,t in enumerate(self.traces):
            traces[i,-len(t):]=t;valid[i,-len(t):]=True
        path=self.output/(self.label+'.npz')
        np.savez_compressed(path,values=traces,valid=valid,env_ids=np.asarray([r['env_id'] for r in self.records]),
            initial_delay20ms=self.initial_delay.cpu().numpy(),failed=self.failed.cpu().numpy())
        meta={'command':self.label,'target':self.target,'num_envs':self.env.num_envs,'step_dt':self.dt,'fields':self.fields,
            'joint_names':self.robot.joint_names,'body_names':self.robot.body_names,
            'soft_joint_limits':self.robot.data.soft_joint_pos_limits[0].cpu().tolist(),
            'hard_joint_limits':self.robot.data.joint_pos_limits[0].cpu().tolist(),
            'records':self.records,'at_risk_step_counts':self.exposure,
            'failed_trials':int(self.failed.sum()),'fall_rate_wilson95':wilson(int(self.failed.sum()),self.env.num_envs),
            'phase_definition':'contact 00 flight,10 left-only,01 right-only,11 double; cycle phase since same-foot contact onset divided by preceding onset period, -1 unavailable',
            'limits_definition':'within0.02rad of configured soft and hard limits stored separately; near-limit proximity is not proof of physical hard-stop contact',
            'window_definition':'up to1s including terminal pre-reset sample; first true termination per environment only'}
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
        return {'trace':str(path),'metadata':str(path.with_suffix('.json')),'failed_trials':len(self.records)}
