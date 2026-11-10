"""Issue12 matched-arm instrumentation, full-state load audit and frozen calibration."""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from unitree_rl_lab.tasks.locomotion.robots.tw.post_push_recovery import TERM, pitch_cost, push_window


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: str(k)):
                h.update(repr(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x: add(v)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def audit_loaded(runner, base, checkpoint, output):
    old = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current = runner.alg.save()
    checks = {}
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        checks[key] = {"checkpoint_sha256": digest(old[key]), "loaded_sha256": digest(current[key])}
        assert checks[key]["checkpoint_sha256"] == checks[key]["loaded_sha256"], key
    assert runner.current_learning_iteration == old["iter"]
    record = {"passed": True, "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "iteration": old["iter"], "exact_state": checks,
              "observation_dims": base.observation_manager.group_obs_dim,
              "actor_parameter_shapes": {k: list(v.shape) for k, v in current["actor_state_dict"].items()},
              "critic_parameter_shapes": {k: list(v.shape) for k, v in current["critic_state_dict"].items()},
              "optimizer_groups": current["optimizer_state_dict"]["param_groups"],
              "std_reset": False}
    Path(output).write_text(json.dumps(record, indent=2))
    return record


class RewardMonitor:
    def __init__(self, base, weight, output):
        self.base, self.weight = base, float(weight)
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "reward_contributions.csv"
        if self.path.exists():
            raise RuntimeError("Refusing to overwrite an existing monitor")
        self.names = ["environment_steps", "transitions", "raw_cost_mean", "active_fraction",
                      "new_reward_per_step", "old_reward_per_step", "total_reward_per_step",
                      "xy_tracking_reward_per_step", "yaw_tracking_reward_per_step"]
        self.fields = ["iteration"] + self.names + ["abs_new_over_abs_old", "abs_new_over_tracking",
                                                   "max_term_identity_error"]
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields, lineterminator="\n")
        self.writer.writeheader(); self.file.flush()
        self.acc = torch.zeros(9, dtype=torch.float64, device=base.device)
        self.error = torch.zeros((), device=base.device)
        self.total_steps = 0
        manager = base.reward_manager
        self.indices = {n: manager.active_terms.index(n) for n in
                        [TERM, "track_lin_vel_xy", "track_ang_vel_z"]}
        original = manager.compute
        def compute(dt):
            result = original(dt)
            raw = pitch_cost(base)
            new = manager._step_reward[:, self.indices[TERM]] * dt
            expected = -self.weight * raw * dt
            self.error.copy_(torch.maximum(self.error, ((new - expected).abs() / (1 + expected.abs())).max()))
            xy = manager._step_reward[:, self.indices["track_lin_vel_xy"]] * dt
            yaw = manager._step_reward[:, self.indices["track_ang_vel_z"]] * dt
            self.acc += torch.stack((result.new_tensor(1), result.new_tensor(base.num_envs),
                         raw.sum(), push_window(base).sum(), new.sum(),
                         (result - new).sum(), result.sum(), xy.sum(), yaw.sum())).double()
            self.total_steps += 1
            return result
        manager.compute = compute

    def flush(self, iteration, tb_writer=None):
        a = self.acc.detach().cpu().tolist()
        assert a[0] > 0 and a[1] > 0
        row = dict(zip(self.names, a[:2] + [x/a[1] for x in a[2:]]))
        row["iteration"] = iteration
        old, new = row["old_reward_per_step"], row["new_reward_per_step"]
        track = row["xy_tracking_reward_per_step"] + row["yaw_tracking_reward_per_step"]
        row["abs_new_over_abs_old"] = abs(new/old) if abs(old) > 1e-12 else None
        row["abs_new_over_tracking"] = abs(new/track) if abs(track) > 1e-12 else None
        row["max_term_identity_error"] = self.error.item()
        assert row["max_term_identity_error"] <= 1e-6
        self.writer.writerow(row); self.file.flush()
        if tb_writer is not None:
            for key, value in row.items():
                if value is not None and math.isfinite(float(value)):
                    tb_writer.add_scalar("Issue12/" + key, value, iteration)
        self.acc.zero_(); self.error.zero_()
        return row

    def attach(self, runner):
        original = runner.logger.log
        def log(*args, **kwargs):
            original(*args, **kwargs)
            self.flush(kwargs["it"], runner.logger.writer)
        runner.logger.log = log

    def close(self):
        self.file.close()


def calibration(runner, env, base, monitor, steps, output):
    """Frozen stochastic parent; no PPO storage, normalizer updates or optimizer steps."""
    if steps != 1024 or base.num_envs != 4096 or monitor.weight != 0:
        raise ValueError("Frozen issue12 calibration requires 1024 steps,4096 envs,weight0")
    before = digest(runner.alg.save())
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    runner.alg.eval_mode()
    obs = env.get_observations().to(runner.device)
    with torch.inference_mode():
        for _ in range(steps):
            actions = runner.alg.actor(obs, stochastic_output=True)
            obs, _, dones, _ = env.step(actions.to(env.device))
            runner.alg.actor.reset(dones)
            runner.alg.critic.reset(dones)
            obs = obs.to(runner.device)
    assert digest(runner.alg.save()) == before, "Calibration changed actor/critic/optimizer state"
    row = monitor.flush(-1); row["step_dt"] = base.step_dt; row["no_optimizer_or_parameter_or_normalizer_updates"] = True
    row["sampling"] = "stochastic frozen loaded actor; original training distribution; random initial episode lengths"
    Path(output).write_text(json.dumps(row, indent=2))


def smoke(runner, env, base, monitor, steps, output):
    import numpy as np
    from trial_risk_records import state_audit
    before = digest(runner.alg.save())
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    runner.alg.eval_mode()
    obs = env.get_observations().to(runner.device)
    rewards, costs = [], []
    original = base.reward_manager.compute
    def capture(dt):
        result = original(dt)
        rewards.append(result.detach().clone())
        costs.append(pitch_cost(base).detach().clone())
        return result
    base.reward_manager.compute = capture
    with torch.inference_mode():
        for _ in range(steps):
            actions = runner.alg.actor(obs, stochastic_output=False)
            obs, _, dones, _ = env.step(actions.to(env.device))
            runner.alg.actor.reset(dones); runner.alg.critic.reset(dones)
            obs = obs.to(runner.device)
    assert digest(runner.alg.save()) == before
    path = Path(output)
    np.savez_compressed(path / 'smoke_rewards.npz', rewards=torch.stack(rewards).cpu().numpy(),
                        costs=torch.stack(costs).cpu().numpy())
    state_audit(base, path / 'final_state', 'final')
    if monitor is not None:
        (path / 'smoke_monitor.json').write_text(json.dumps(monitor.flush(-2), indent=2))
    (path / 'smoke_complete').write_text('No optimizer or model state changes; deterministic parent rollout.')
