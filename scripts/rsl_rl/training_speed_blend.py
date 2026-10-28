"""Opt-in environment transform matching deployed reference-speed blend.
PPO retains the student's latent action and original log probability.
The mixed action then enters the existing ankle filter. No new observations.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from reference_speed_blend import candidate_weight,mix_actions


def make_contract(teacher_export,teacher_checkpoint,action_names=None,action_scope='all',controlled_likelihood=None):
    export=Path(teacher_export);checkpoint=Path(teacher_checkpoint)
    contract={'version':1,'teacher_export':str(export.resolve()),
        'teacher_export_sha256':hashlib.sha256(export.read_bytes()).hexdigest(),
        'teacher_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'maximum_candidate_weight':1.,'blend_speed_low':.05,'blend_speed_high':.1,
        'order':'reference speed mix then existing ankles','ankle_alpha':.5,
        'teacher_input':'same current original policy observation',
        'PPO_action':'raw student sample with original student log probability',
        'release':False,'trace_prefix_steps':64,'teacher_matmul':'float32_no_tf32'}
    if action_scope!='all':
        from candidate_scope_contract import add_training_scope
        contract=add_training_scope(contract,action_names,action_scope)
    if controlled_likelihood is not None:
        from controlled_action_likelihood import attach_training_contract
        contract=attach_training_contract(contract,controlled_likelihood)
    return contract


def audit_teacher_binding(teacher_export,teacher_checkpoint):
    actor=torch.load(teacher_checkpoint,map_location='cpu',weights_only=False)['actor_state_dict']
    exported=torch.jit.load(str(teacher_export),map_location='cpu').state_dict()
    if set(actor)-set(exported)!={'distribution.std_param'} or not set(exported)<=set(actor):
        raise ValueError('Mixture teacher export does not contain the checkpoint actor')
    checks={k:torch.equal(actor[k],v) for k,v in exported.items()}
    if not checks or not all(checks.values()):raise ValueError('Mixture teacher weights differ from frozen anchor')
    if actor['mlp.0.weight'].shape[1]!=195:raise ValueError('Expected original teacher observation ABI195')
    return {'passed':True,'exact_parameter_checks':checks,'teacher_input_dim':195}


def validate_resume(saved,current):
    if saved!=current:raise ValueError('Resume must preserve training reference-speed mixture contract')


def validate_checkpoint_jobs(checkpoint,jobs):
    p=Path(checkpoint).parent/'params/training_speed_blend.json'
    if not p.exists():return None
    return validate_jobs(json.loads(p.read_text()),jobs)


def validate_jobs(c,jobs):
    if not jobs:raise ValueError('Training mixture requires explicit batch evaluation')
    original_contract=c
    from candidate_scope_contract import validate_scope_jobs
    c,scope=validate_scope_jobs(c,jobs)
    if c.get('version')!=1 or c.get('release') is not False or c.get('order')!='reference speed mix then existing ankles':
        raise ValueError('Unsupported training mixture contract')
    if hashlib.sha256(Path(c['teacher_export']).read_bytes()).hexdigest()!=c['teacher_export_sha256']:
        raise ValueError('Frozen mixture teacher export changed')
    if (c['maximum_candidate_weight'],c['blend_speed_low'],c['blend_speed_high'],c['ankle_alpha'])!=(1.,.05,.1,.5):
        raise ValueError('Unsupported trained mixture parameters')
    forbidden=('blend_teacher_export','candidate_hip_filter_alpha','candidate_leg_filter_alpha','blend_release_tau','locomotion_damping_gain')
    for j in jobs:
        if j.get('controller','policy')!='policy' or any(k in j for k in forbidden):
            raise ValueError('Conflicting trained mixture controller or transform')
        if 'reference_speed_teacher_export' not in j:raise ValueError('Trained mixture replay requires frozen teacher')
        if hashlib.sha256(Path(j['reference_speed_teacher_export']).read_bytes()).hexdigest()!=c['teacher_export_sha256']:
            raise ValueError('Replay teacher differs from training')
        for k in ('maximum_candidate_weight','blend_speed_low','blend_speed_high'):
            if j.get(k)!=c[k]:raise ValueError('Replay mixture parameters differ from training')
        if j.get('leg_filter_alpha')!=c['ankle_alpha'] or j.get('leg_filter_scope')!='ankles' or j.get('leg_filter_mode','absolute')!='absolute':
            raise ValueError('Replay ankle filter differs from training')
    return {'contract':original_contract,'matched_batch_jobs':len(jobs)}


class TrainingSpeedBlend:
    def __init__(self,env,teacher,command,contract,trace_steps=64,action_names=None):
        self.env=env;self.teacher=teacher.eval()
        for parameter in self.teacher.parameters():parameter.requires_grad_(False)
        self.command=command;self.index=command.cfg.body_names.index('base_link')
        self.contract=contract;self.trace_steps=trace_steps;self.current=None
        self.action_scope=None
        if 'candidate_action_scope' in contract:
            from candidate_scope_contract import require_scope_action_names
            if action_names is None:raise ValueError('Training action scope requires actual joint order')
            self.action_scope=require_scope_action_names(contract,action_names)
        self.trace=[];self.completed_steps=0;self.active_samples=0;self.zero_samples=0
    def __getattr__(self,name):return getattr(self.env,name)
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):self.env.episode_length_buf=value
    def _cache(self,obs):
        base=obs.get('contact_base_policy',obs['policy'])
        if base.ndim!=2 or base.shape[0]!=self.num_envs:raise ValueError('Expected flat same-step teacher observations')
        self.current=base.detach().clone();return obs
    def get_observations(self):return self._cache(self.env.get_observations())
    def reset(self):
        self.current=None
        obs,extras=self.env.reset();return self._cache(obs),extras
    @torch.inference_mode()
    def step(self,actions):
        if self.current is None:raise RuntimeError('Get current observations before training mixture step')
        speed=torch.linalg.vector_norm(self.command.body_lin_vel_w[:,self.index,:2],dim=-1)
        weight=candidate_weight(speed,1.,.05,.1)
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            teacher=self.teacher(self.current)
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous_tf32
        if self.action_scope is None:mixed=mix_actions(teacher,actions,weight)
        else:
            from candidate_action_scope import mix_scoped_actions
            mixed=mix_scoped_actions(teacher,actions,weight,self.action_scope['scope'],self.action_scope['action_joint_names'])
        capture=len(self.trace)<self.trace_steps
        record=None
        if capture:
            record={k:v.detach().cpu().numpy().copy() for k,v in
                {'teacher_observation':self.current,'raw_action':actions,'teacher_action':teacher,
                 'reference_speed':speed,'candidate_weight':weight,'mixed_action':mixed}.items()}
        obs,rewards,dones,extras=self.env.step(mixed)
        if capture:
            record['terminal']=dones.detach().cpu().numpy().copy();self.trace.append(record)
        self.completed_steps+=1;self.active_samples+=int((weight>0).sum());self.zero_samples+=int((weight==0).sum())
        return self._cache(obs),rewards,dones,extras
    def save_trace(self,path):
        if not self.trace:raise RuntimeError('No actual mixture steps captured')
        np.savez_compressed(path,**{k:np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
        return {'completed_steps':self.completed_steps,'captured_steps':len(self.trace),
            'active_samples':self.active_samples,'zero_samples':self.zero_samples,
            'num_envs':self.num_envs,'contract':self.contract}
