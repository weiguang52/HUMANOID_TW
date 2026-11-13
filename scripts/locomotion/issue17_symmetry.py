"""Training-only spatial reflection plus half-cycle phase; NOT issue16 rollout mirror."""
import torch
from issue16_mirror import Mirror101

class TrainingMirror:
    def __init__(self,joint_names,action_names=None,action_scales=None,device='cpu'):
        self.spatial=Mirror101(joint_names,action_names,action_scales,device=device)
    def actor(self,x):
        assert x.shape[-1]==101
        y=self.spatial.observation(x)
        y[...,99:101]=-x[...,99:101]
        return y
    def critic(self,x):
        assert x.shape[-1]==134
        y=x.clone()
        y[...,:3]=x[...,:3]*x.new_tensor([1,-1,1])
        y[...,3:6]=x[...,3:6]*x.new_tensor([-1,1,-1])
        y[...,6:9]=x[...,6:9]*x.new_tensor([1,-1,1])
        y[...,9:12]=x[...,9:12]*x.new_tensor([1,-1,-1])
        for start in [12,42,72,102]:
            y[...,start:start+30]=x[...,start:start+30].index_select(-1,self.spatial.perm)*self.spatial.sign
        y[...,132:134]=-x[...,132:134]
        return y
    def action(self,x):return self.spatial.action(x)

def mapping_for_env(env):
    base=env.unwrapped
    if not hasattr(base,'_issue17_training_mirror'):
        robot=base.scene['robot'];term=base.action_manager.get_term('JointPositionAction')
        actor=['base_ang_vel','projected_gravity','velocity_commands','joint_pos_rel','joint_vel_rel','last_action','gait_phase']
        critic=['base_lin_vel','base_ang_vel','projected_gravity','velocity_commands','joint_pos_rel','joint_vel_rel','joint_effort','last_action','gait_phase']
        assert base.observation_manager.active_terms['policy']==actor
        assert base.observation_manager.active_terms['critic']==critic
        names=list(term._joint_names)
        assert names==list(robot.joint_names),'Exact TW action/observation ordering required'
        scales=term._scale
        if isinstance(scales,(int,float)):scales=torch.full((30,),float(scales),device=base.device)
        elif scales.ndim==2:
            assert torch.equal(scales,scales[:1].expand_as(scales));scales=scales[0]
        m=TrainingMirror(robot.joint_names,names,scales,device=base.device)
        assert torch.equal(m.spatial.action_factor.abs(),torch.ones_like(m.spatial.action_factor)),'Last-action features require equal reflected scales'
        defaults=robot.data.default_joint_pos
        assert torch.allclose(defaults,defaults[:,m.spatial.perm]*m.spatial.sign,atol=1e-6,rtol=0)
        base._issue17_training_mirror=m
    return base._issue17_training_mirror

def data_augmentation_func(env,obs=None,actions=None):
    """RSL5 TensorDict API: originals first, then one transformed copy; no RNG draws."""
    m=mapping_for_env(env)
    out_obs=None;out_actions=None
    if obs is not None:
        assert set(obs.keys())=={'policy','critic'}
        mirrored=obs.clone()
        mirrored['policy']=m.actor(obs['policy']);mirrored['critic']=m.critic(obs['critic'])
        out_obs=torch.cat([obs,mirrored],dim=0)
    if actions is not None:out_actions=torch.cat([actions,m.action(actions)],dim=0)
    base=env.unwrapped
    if obs is not None and actions is not None:
        audit=getattr(base,'_issue17_augmentation_audit',None)
        if audit is None:
            audit=dict(training_batch_calls=0,original_batch_sizes=[],augmented_batch_sizes=[])
            base._issue17_augmentation_audit=audit
        audit['training_batch_calls']+=1
        n=int(obs.batch_size[0]);size=int(out_obs.batch_size[0])
        assert size==2*n
        if n not in audit['original_batch_sizes']:audit['original_batch_sizes'].append(n)
        if size not in audit['augmented_batch_sizes']:audit['augmented_batch_sizes'].append(size)
    return out_obs,out_actions
