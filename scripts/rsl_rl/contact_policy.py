"""Opt-in contact input and explicit original-policy checkpoint migration.

Enabled only by explicit training flag or saved checkpoint contract.
Existing policies keep their input ABI.
"""
import copy
import torch
from contact_feedback import ground_contact_features
BASE_POLICY='contact_base_policy'
EXTRA_DIM=8

class ContactPolicyObservations:
    def __init__(self,env,mode,sensor_prefix='slip_'):
        if mode not in ['actual','zeros']:raise ValueError('Contact mode must be actual or zeros')
        self.env=env;self.mode=mode;self.sensor_prefix=sensor_prefix;self.base_dim=None
    def __getattr__(self,name):return getattr(self.env,name)
    @property
    def episode_length_buf(self):return self.env.episode_length_buf
    @episode_length_buf.setter
    def episode_length_buf(self,value):self.env.episode_length_buf=value
    @torch.inference_mode()
    def _pack(self,obs):
        if BASE_POLICY in obs:raise ValueError('Contact observations already present')
        base=obs['policy']
        if base.ndim!=2 or base.shape[0]!=self.num_envs:raise ValueError('Expected flat policy observations')
        if self.base_dim is None:self.base_dim=base.shape[-1]
        if base.shape[-1]!=self.base_dim:raise ValueError('Original observation dimension changed')
        if self.mode=='actual':extra=ground_contact_features(self.env.unwrapped,self.sensor_prefix)
        else:extra=base.new_zeros(self.num_envs,EXTRA_DIM)
        if extra.shape!=(self.num_envs,EXTRA_DIM):raise ValueError('Contact feature dimension changed')
        out=obs.clone(False);out[BASE_POLICY]=base.clone();out['policy']=torch.cat([base,extra],dim=-1)
        return out
    def get_observations(self):return self._pack(self.env.get_observations())
    def reset(self):
        obs,extras=self.env.reset();return self._pack(obs),extras
    def step(self,actions):
        obs,rewards,dones,extras=self.env.step(actions);return self._pack(obs),rewards,dones,extras


def expand_actor_state(actor,state,extra_dim=EXTRA_DIM):
    """Pad only input columns; every other key/shape must match exactly."""
    if actor.obs_normalization:raise ValueError('Contact prototype requires original identity observation normalizer')
    if actor.obs_groups!=['policy']:raise ValueError('Expected policy-only actor')
    first=next((name for name,m in actor.named_modules() if isinstance(m,torch.nn.Linear)),None)
    if first is None:raise ValueError('No actor input linear layer')
    key=first+'.weight';target=actor.state_dict()
    if state.keys()!=target.keys():raise ValueError('Original actor parameter keys differ')
    for k in state:
        if k!=key and state[k].shape!=target[k].shape:raise ValueError('Non-input parameter dimensions changed')
    old=state[key];new=target[key]
    out=copy.deepcopy(state)
    if old.shape==new.shape:return out,{'expanded':False,'key':key,'input_dim':new.shape[1]}
    if old.ndim!=2 or new.shape!=(old.shape[0],old.shape[1]+extra_dim):
        raise ValueError('Expected exactly eight added actor input columns')
    out[key]=torch.cat([old,old.new_zeros(old.shape[0],extra_dim)],dim=1)
    return out,{'expanded':True,'key':key,'base_dim':old.shape[1],'input_dim':new.shape[1],'extra_columns_zero':True}


def migrate_checkpoint(alg,checkpoint):
    """Preserve Adam history by padding corresponding moments with zero columns.

    Uses the current optimizer parameter ordering and verifies all state shapes.
    No new parameters or parameter groups may be introduced by this migration.
    """
    if alg.rnd or alg.symmetry or alg.is_multi_gpu:raise ValueError('Audited only for plain feed-forward PPO')
    loaded=copy.deepcopy(checkpoint);loaded['actor_state_dict'],audit=expand_actor_state(alg.actor,loaded['actor_state_dict'])
    groups=loaded['optimizer_state_dict']['param_groups'];current=alg.optimizer.param_groups
    if len(groups)!=len(current):raise ValueError('Optimizer groups changed')
    padded=0;input_param=dict(alg.actor.named_parameters())[audit['key']]
    for saved,live in zip(groups,current):
        if len(saved['params'])!=len(live['params']):raise ValueError('Optimizer parameter count changed')
        for saved_id,param in zip(saved['params'],live['params']):
            slot=loaded['optimizer_state_dict']['state'].get(saved_id,{})
            for name,value in list(slot.items()):
                if not isinstance(value,torch.Tensor) or value.ndim==0:continue
                if value.shape==param.shape:continue
                if not audit['expanded'] or param is not input_param or value.shape!=(param.shape[0],param.shape[1]-EXTRA_DIM):
                    raise ValueError('Unexpected optimizer state shape change')
                slot[name]=torch.cat([value,value.new_zeros(value.shape[0],EXTRA_DIM)],dim=1);padded+=1
    audit['optimizer_moment_tensors_padded']=padded
    return loaded,audit


def load_contact_checkpoint(runner,path):
    loaded=torch.load(path,weights_only=False,map_location=runner.alg.device)
    loaded,audit=migrate_checkpoint(runner.alg,loaded)
    if runner.alg.load(loaded,None,True):runner.current_learning_iteration=loaded['iter']
    audit['iteration']=runner.current_learning_iteration
    return audit


def validate_contact_contract(contract):
    expected={'version':1,'extra_dim':8,'load_scale_N':100.,'speed_scale_m_s':.1,'loaded_threshold_N':5.,'reset':'zero first per-environment observation','sensor_prefix_training':'slip_','teacher_input':'original policy only'}
    if not isinstance(contract,dict) or any(contract.get(k)!=v for k,v in expected.items()):
        raise ValueError('Unsupported contact input normalization or reset contract')
    if contract.get('mode') not in ['actual','zeros'] or not isinstance(contract.get('base_dim'),int) or contract['base_dim']<=0:
        raise ValueError('Invalid contact mode or original input dimension')
    return contract
