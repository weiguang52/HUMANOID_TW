"""Full-state critic-only input expansion without extra random initialization."""
import copy,json
from pathlib import Path
import torch
from issue12_monitor import digest

def construct_parent_width_runner(env,cfg,log_dir,device):
    from rsl_rl.runners import OnPolicyRunner
    original=env.get_observations
    def parent_observations():
        obs=original()
        assert obs["policy"].shape[-1]==101 and obs["critic"].shape[-1]==136
        old=obs.clone(False);old["critic"]=obs["critic"][...,:134]
        return old
    env.get_observations=parent_observations
    try:return OnPolicyRunner(env,cfg,log_dir=log_dir,device=device)
    finally:env.get_observations=original

def expand_loaded_critic(runner,base,output):
    """Called AFTER ordinary strict parent load and exact parent-state audit."""
    alg=runner.alg;critic=alg.critic
    assert critic.obs_dim==134 and not critic.obs_normalization
    assert alg.actor.obs_dim==101 and critic.obs_groups==["critic"]
    assert tuple(base.observation_manager.group_obs_dim["critic"])==(136,)
    before=copy.deepcopy(alg.save());rng=torch.get_rng_state().clone()
    cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
    obs=base.obs_buf
    actor_before=alg.actor(obs).detach().clone()
    first=critic.mlp[0];old=first.weight
    assert old.shape[1]==134 and old.grad is None
    new=torch.nn.Parameter(torch.cat((old.detach(),old.new_zeros((old.shape[0],2))),dim=1))
    state=alg.optimizer.state.pop(old)
    migrated={}
    for key,value in state.items():
        if isinstance(value,torch.Tensor) and value.shape==old.shape:
            migrated[key]=torch.cat((value,value.new_zeros((value.shape[0],2))),dim=1)
        else:migrated[key]=value
    count=0
    for group in alg.optimizer.param_groups:
        for i,p in enumerate(group["params"]):
            if p is old:group["params"][i]=new;count+=1
    assert count==1
    alg.optimizer.state[new]=migrated
    first.weight=new;first.in_features=136;critic.obs_dim=136
    stored=alg.storage.observations["critic"]
    assert stored.shape[-1]==134 and alg.storage.step==0
    alg.storage.observations["critic"]=torch.cat((stored,stored.new_zeros((*stored.shape[:-1],2))),dim=-1)
    after=alg.save()
    assert digest(before["actor_state_dict"])==digest(after["actor_state_dict"])
    stripped=copy.deepcopy(after["critic_state_dict"]);key="mlp.0.weight"
    assert torch.count_nonzero(stripped[key][:,134:])==0
    stripped[key]=stripped[key][:,:134].contiguous()
    assert digest(stripped)==digest(before["critic_state_dict"])
    old_opt=before["optimizer_state_dict"];new_opt=copy.deepcopy(after["optimizer_state_dict"]);padded=[]
    for pid,terms in new_opt["state"].items():
        for key,value in terms.items():
            prior=old_opt["state"][pid][key]
            if isinstance(value,torch.Tensor) and value.ndim==2 and value.shape[-1]==136:
                assert prior.shape[-1]==134 and torch.count_nonzero(value[:,134:])==0
                terms[key]=value[:,:134].contiguous();padded.append([pid,key])
    assert len(padded)==2 and digest(new_opt)==digest(old_opt)
    assert torch.equal(actor_before,alg.actor(obs))
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    result={"passed":True,"critic_before":134,"critic_after":136,"policy":101,"zero_new_weight_columns":True,"only_two_Adam_moment_columns_padded":padded,
            "all_old_actor_critic_optimizer_state_exact":True,"all_optimizer_steps_and_parameter_groups_preserved":True,
            "actor_output_same_observation_bitwise_equal":True,"zero_random_numbers_consumed":True,"std_reset":False,
            "expanded_critic_shapes":{k:list(v.shape) for k,v in after["critic_state_dict"].items()}}
    Path(output).write_text(json.dumps(result,indent=2))
    return result
