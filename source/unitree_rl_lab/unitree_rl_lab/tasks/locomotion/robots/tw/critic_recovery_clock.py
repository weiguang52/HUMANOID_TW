"""Opt-in privileged critic clock sharing the exact existing reward timer."""
import torch
from .post_push_recovery import push_clock,push_window,reset_push_clock,mark_push

def recovery_clock(env,window_s=1.0):
    active=push_window(env,window_s)
    steps=round(window_s/env.step_dt)
    age=env.common_step_counter-push_clock(env)
    remaining=(1.0-age.to(torch.float32)/steps).clamp(0.,1.)*active
    return torch.stack((active.to(torch.float32),remaining),dim=-1)

def clocked_push(env,env_ids,velocity_range,original_push,asset_cfg=None):
    """Delegate the original evaluation sampler once, then mark the same clock."""
    if asset_cfg is None:original_push(env,env_ids,velocity_range)
    else:original_push(env,env_ids,velocity_range,asset_cfg)
    mark_push(env,env_ids)

def configure_critic_recovery_clock(cfg,enabled=False,evaluation=False):
    if not enabled:return
    from isaaclab.managers import ObservationTermCfg,EventTermCfg
    if getattr(cfg.observations.critic,"recovery_clock",None) is not None:
        raise ValueError("Critic clock already configured")
    if not evaluation and getattr(cfg.events,"reset_post_push_clock",None) is None:
        raise ValueError("Training critic clock requires the original post-push reward configuration")
    cfg.observations.critic.recovery_clock=ObservationTermCfg(func=recovery_clock,params={"window_s":1.0})
    if evaluation:
        cfg.events.reset_post_push_clock=EventTermCfg(func=reset_push_clock,mode="reset")
        if cfg.events.push_robot is not None:
            event=cfg.events.push_robot
            assert "original_push" not in event.params
            event.params={**event.params,"original_push":event.func}
            event.func=clocked_push
