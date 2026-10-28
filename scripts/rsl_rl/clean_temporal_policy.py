"""Opt-in clean critic reconstruction for temporal loss only; no NN ABI change."""
import math
import torch
from second_order_anchor import PreviousTwoPoliciesTraining
CLEAN='clean_temporal_policy'
PAIRS=(('motion_anchor_ori_b','motion_anchor_ori_b'),('base_lin_vel','base_lin_vel'),
       ('base_ang_vel','base_ang_vel'),('joint_pos_rel','joint_pos'),('joint_vel_rel','joint_vel'))

def canonical(value):
    if value is None or isinstance(value,(str,int,bool)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite observation parameter')
        return value
    if isinstance(value,slice):return {'slice':[value.start,value.stop,value.step]}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,dict):return {k:canonical(v) for k,v in sorted(value.items())}
    if hasattr(value,'__dict__'):return canonical(vars(value))
    raise ValueError('Unsupported observation parameter type '+type(value).__name__)

def make_contract(policy,critic,policy_dim=203,critic_dim=315):
    def layout(terms):
        out={};offset=0
        for t in terms:
            if t['name'] in out or isinstance(t['dim'],bool) or not isinstance(t['dim'],int) or t['dim']<=0:raise ValueError('Invalid observation layout')
            out[t['name']]={**t,'range':[offset,offset+t['dim']]};offset+=t['dim']
        return out,offset
    p,pdim=layout(policy);c,cdim=layout(critic)
    if (pdim,policy_dim,cdim,critic_dim)!=(195,203,315,315):raise ValueError('Expected original195+contact8 actor and315 critic')
    expected={a for a,_ in PAIRS}
    if {n for n,t in p.items() if t['noise'] is not None}!=expected:raise ValueError('Unexpected policy noise terms')
    if any(t['noise'] is not None for t in c.values()):raise ValueError('Critic must be clean')
    rows=[]
    for a,b in PAIRS:
        if a not in p or b not in c:raise ValueError('Missing same-state clean term')
        x,y=p[a],c[b]
        if x['dim']!=y['dim'] or x['func']!=y['func'] or x['params']!=y['params']:raise ValueError('Actor/critic term semantics differ')
        for t in [x,y]:
            if any(t[k] is not None for k in ['scale','clip','modifiers']) or t['history_length']!=0:raise ValueError('Term transforms/history unsupported')
        noise=x['noise']
        if noise['func']!='isaaclab.utils.noise.noise_model:uniform_noise' or noise['operation']!='add':raise ValueError('Expected additive uniform training noise')
        if not all(isinstance(noise[k],(int,float)) and math.isfinite(noise[k]) for k in ['n_min','n_max']) or noise['n_min']>noise['n_max']:raise ValueError('Invalid noise bounds')
        rows.append({'policy_term':a,'critic_term':b,'policy_range':x['range'],'critic_range':y['range'],'func':x['func'],'params':x['params'],'noise':noise})
    return {'version':1,'kind':'same_state_clean_critic_for_temporal_only','policy_dim':203,'base_policy_dim':195,'critic_dim':315,'replacements':rows,'policy_and_teacher_anchor':'original noisy observations unchanged','PPO':'original noisy policy unchanged','actor_critic_groups':'unchanged; clean key auxiliary only','history':'current and two previous clean observations; exclude first two actions after each env reset','rng':'no random sampling or extra observation/environment compute'}

def reconstruct(obs,contract):
    if contract.get('version')!=1 or contract.get('kind')!='same_state_clean_critic_for_temporal_only':raise ValueError('Unsupported clean temporal contract')
    p,c=obs['policy'],obs['critic']
    if p.ndim!=2 or c.ndim!=2 or p.shape[0]!=c.shape[0] or p.shape[1]!=contract['policy_dim'] or c.shape[1]!=contract['critic_dim']:raise ValueError('Clean reconstruction dimensions changed')
    if p.dtype!=c.dtype or p.device!=c.device or not p.is_floating_point():raise ValueError('Observation dtype/device mismatch')
    out=p.clone()
    for r in contract['replacements']:
        a,z=r['policy_range'];i,j=r['critic_range']
        if z-a!=j-i or not (0<=a<z<=195 and 0<=i<j<=315):raise ValueError('Invalid reconstruction ranges')
        out[:,a:z]=c[:,i:j]
    return out

def contract_from_manager(manager):
    def function_name(f):return f.__module__+':'+f.__name__
    if not manager.cfg.policy.enable_corruption or manager.cfg.critic.enable_corruption:raise ValueError('Expected noisy policy and clean critic')
    groups=[]
    for g in ['policy','critic']:
        if not manager.group_obs_concatenate[g]:raise ValueError('Flat observation groups required')
        names=manager.active_terms[g];dims=manager.group_obs_term_dim[g];cfgs=manager._group_obs_term_cfgs[g]
        if not len(names)==len(dims)==len(cfgs):raise ValueError('Observation manager layout inconsistent')
        rows=[]
        for name,dim,cfg in zip(names,dims,cfgs):
            if len(dim)!=1:raise ValueError('Flat observation terms required')
            n=cfg.noise;noise=None if n is None else {'func':function_name(n.func),'operation':n.operation,'n_min':float(n.n_min),'n_max':float(n.n_max)}
            rows.append({'name':name,'dim':int(dim[0]),'func':function_name(cfg.func),'params':canonical(cfg.params),'noise':noise,'scale':canonical(cfg.scale),'clip':canonical(cfg.clip),'modifiers':canonical(cfg.modifiers),'history_length':cfg.history_length})
        groups.append(rows)
    return make_contract(*groups)

class CleanPolicyObservations:
    def __init__(self,env,contract=None):
        self.env=env;self.contract=contract_from_manager(env.unwrapped.observation_manager) if contract is None else contract
    def __getattr__(self,name):return getattr(self.env,name)
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):self.env.episode_length_buf=value
    @torch.inference_mode()
    def _pack(self,obs):
        if CLEAN in obs:raise ValueError('Clean temporal observations already installed')
        out=obs.clone(False);out[CLEAN]=reconstruct(obs,self.contract);return out
    def get_observations(self):return self._pack(self.env.get_observations())
    def reset(self):
        obs,extras=self.env.reset();return self._pack(obs),extras
    def step(self,actions):
        obs,rewards,dones,extras=self.env.step(actions);return self._pack(obs),rewards,dones,extras

class CleanTwoPoliciesTraining(PreviousTwoPoliciesTraining):
    def _pack(self,obs,previous=None,older=None,count=None):
        if CLEAN not in obs:raise ValueError('Clean current observation missing')
        original=obs['policy'];view=obs.clone(False);view['policy']=obs[CLEAN]
        out=super()._pack(view,previous,older,count);out['policy']=original
        return out

def validate_resume_source(saved,requested):
    if requested not in ['policy','clean_critic']:raise ValueError('Unknown temporal observation source')
    prior=saved.get('temporal_observation_source','policy')
    if prior not in ['policy','clean_critic']:raise ValueError('Unknown saved temporal observation source')
    if saved.get('temporal_coefficient',0) and prior!=requested:raise ValueError('Resume must preserve temporal observation source')

def configure_clean_temporal(alg,contract):
    from second_order_anchor import PREVIOUS,OLDER,VALID
    if alg.anchor_temporal_order!=2 or alg.anchor_temporal_target!='student_mean' or alg.anchor_temporal_coefficient<=0:raise ValueError('Clean temporal requires positive student mean order2')
    if alg.actor.obs_groups!=['policy'] or alg.critic.obs_groups!=['critic'] or alg.actor.obs_normalization:raise ValueError('Expected unchanged identity policy/critic groups')
    stored=alg.storage.observations
    for key in [CLEAN,PREVIOUS,OLDER]:
        if key not in stored or stored[key].shape[-1]!=contract['policy_dim']:raise ValueError('Clean current/history must exist before rollout storage construction')
    if VALID not in stored or stored[VALID].shape[-1]!=1:raise ValueError('Clean history valid mask missing')
    if any(k in alg.actor.obs_groups+alg.critic.obs_groups for k in [CLEAN,PREVIOUS,OLDER,VALID]):raise ValueError('Auxiliary clean keys cannot enter NN groups')
    alg.anchor_temporal_observation_source='clean_critic'
    return {'enabled':True,'temporal_observation_source':'clean_critic','contract':contract,'PPO_and_anchor_input_unchanged':True,'all_three_temporal_inputs_clean':True}
