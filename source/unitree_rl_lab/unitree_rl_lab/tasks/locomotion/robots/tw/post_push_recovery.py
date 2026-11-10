"""Opt-in post-push pitch recovery; privileged timing never enters observations."""
import math
import torch

TERM = "post_push_pitch_recovery"
_CLOCK = "_post_push_pitch_step"


def push_clock(env):
    clock = getattr(env, _CLOCK, None)
    if clock is None:
        clock = torch.full((env.num_envs,), -(2**60), dtype=torch.long, device=env.device)
        setattr(env, _CLOCK, clock)
    return clock


def mark_push(env, env_ids):
    push_clock(env)[slice(None) if env_ids is None else env_ids] = env.common_step_counter


def reset_push_clock(env, env_ids=None):
    push_clock(env)[slice(None) if env_ids is None else env_ids] = -(2**60)


def push_window(env, window_s=1.0):
    steps = round(window_s / env.step_dt)
    if steps < 1 or not math.isclose(steps * env.step_dt, window_s, abs_tol=1e-9):
        raise ValueError("Window must be an integral positive number of control steps")
    age = env.common_step_counter - push_clock(env)
    return (age >= 0) & (age <= steps)


def pitch_cost(env, window_s=1.0):
    """ZYX Euler pitch and its exact Euler derivative, in radians and rad/s."""
    data = env.scene["robot"].data
    w, x, y, z = data.root_quat_w.unbind(-1)
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    # For ZYX Euler angles, pitch_dot = cos(roll)*omega_y - sin(roll)*omega_z.
    pitch_rate = roll.cos() * data.root_ang_vel_b[:, 1] - roll.sin() * data.root_ang_vel_b[:, 2]
    return (pitch.square() + 0.1 * pitch_rate.square()) * push_window(env, window_s)


def timed_push(env, env_ids, velocity_range, asset_cfg=None):
    from isaaclab.envs.mdp.events import push_by_setting_velocity
    from isaaclab.managers import SceneEntityCfg
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    # Exactly the original sampler and physics operation, called exactly once.
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    mark_push(env, env_ids)


def configure_post_push_pitch_reward(env_cfg, weight):
    """None: unchanged production defaults. Explicit 0/w: matched experiment arms."""
    if weight is None:
        return
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("The opt-in penalty magnitude must be finite and nonnegative")
    from isaaclab.managers import EventTermCfg, RewardTermCfg
    if env_cfg.events.push_robot is None:
        raise ValueError("Post-push recovery requires the original push event")
    env_cfg.events.push_robot.func = timed_push
    env_cfg.events.reset_post_push_clock = EventTermCfg(func=reset_push_clock, mode="reset")
    setattr(env_cfg.rewards, TERM, RewardTermCfg(func=pitch_cost, weight=-float(weight),
                                              params={"window_s": 1.0}))
