"""Explicit real-feedback diagnostic using the existing SuperMimic policy modules."""

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from supermimic.algo.state import encoder_bias_from_context
from supermimic.algo.trainer import _squeeze
from tensordict import TensorDict
from torch import nn

KIND = "real_physics_ppo_diagnostic_policy_v1"


class ActorAdapter(nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.encoder = ac.encoder
        self.decoder = ac.decoder
        self.obs_norm = ac.obs_norm
        self.goal_norm = ac.goal_norm
        self.obs_dim = ac.cfg.obs_dim
        self.history_len = ac.cfg.history_len

    def forward(self, x):
        obs = x[:, : self.obs_dim].reshape(len(x), self.history_len, -1)
        goal = x[:, self.obs_dim :]
        history = self.obs_norm(obs).reshape(len(x), -1)
        z = torch.tanh(self.encoder(self.goal_norm(goal)))
        return self.decoder(torch.cat([z, history], dim=-1))


class CriticAdapter(nn.Module):
    def __init__(self, ac):
        super().__init__()
        self.critic = ac.critic
        self.norm = ac.critic_norm

    def forward(self, x):
        return self.critic(self.norm(x))


def construct_ppo(ac, obs, num_envs, gamma, noise_std):
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        30,
        hidden_dims=(1,),
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": noise_std,
            "learn_std": False,
        },
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(1,))
    actor.mlp = ActorAdapter(ac)
    critic.mlp = CriticAdapter(ac)
    all_params = list(actor.parameters()) + list(critic.parameters())
    assert len({id(x) for x in all_params}) == len(all_params)
    required = {
        id(x)
        for name, x in ac.named_parameters()
        if any(name.startswith(p) for p in ["encoder.", "decoder.", "critic."])
    }
    assert {id(x) for x in all_params if x.requires_grad} == required
    storage = RolloutStorage("rl", num_envs, 32, obs, [30], device="cuda:0")
    alg = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="fixed",
        clip_param=0.2,
        gamma=gamma,
        lam=0.95,
        entropy_coef=0.0,
        max_grad_norm=1.0,
        device="cuda:0",
    )
    return alg


def bootstrap_timeouts(reward, terminated, truncated, terminal_value, gamma):
    return reward + gamma * terminal_value * (truncated & ~terminated).to(reward.dtype)


class RealFeedbackCollector:
    def __init__(self, trainer):
        self.t = trainer
        self.done_count = 0
        self.timeout_count = 0
        self.reset_live_max_error = 0.0
        self.rewards = []
        self.completed_seconds = []

    @torch.no_grad()
    def observe(self, ids=None):
        t = self.t
        if ids is None:
            ids = torch.arange(t.env.num_envs, device=t.device)
        state = t.extractor.state()[ids]
        context = t.extractor.context()[ids]
        proprio = t.proprio(
            state,
            t.last_action[ids],
            encoder_bias_from_context(context, t.spec.num_joints),
            add_noise=True,
        )
        history = torch.cat([t.history[ids, 1:], proprio[:, None]], dim=1)
        t.history[ids] = history
        goal = t.command.goal()[ids]
        return TensorDict(
            {
                "actor": torch.cat([history.reshape(len(ids), -1), goal], dim=-1),
                "critic": torch.cat([state, goal, context], dim=-1),
            },
            batch_size=[len(ids)],
        )

    @torch.no_grad()
    def step(self, action, alg):
        t = self.t
        clip = t.command.clip_ids.clone()
        times = t.command.time_steps.clone()
        previous = t.last_action.clone()
        _, _, terminated, truncated, _ = t.env.step(action)
        next_state = t.extractor.state().clone()
        reference = _squeeze(t.library.gather(clip, times, 1))
        reward, _ = t.reward(next_state, action, previous, reference)
        assert torch.isfinite(reward).all()
        t.last_action = action.detach().clone()
        idx = t.episode_len
        assert bool((idx < t.episode_rewards.shape[1]).all())
        t.episode_rewards[torch.arange(t.env.num_envs), idx] = reward
        t.episode_len += 1
        t.episode_return += reward
        t.episode_length += 1
        t.env_steps += t.env.num_envs
        next_obs = self.observe()
        done = terminated | truncated
        timeout = truncated & ~terminated
        terminal_value = alg.critic(next_obs).squeeze(-1)
        learning_reward = bootstrap_timeouts(
            reward, terminated, truncated, terminal_value, t.cfg.gamma
        )
        self.timeout_count += int(timeout.sum())
        self.done_count += int(done.sum())
        self.rewards.append(float(reward.mean()))
        if bool(done.any()):
            ids = done.nonzero().flatten()
            live = ~done
            live_context = t.extractor.context()[live].clone()
            clock = t.command.time_steps[live].clone()
            self.completed_seconds.extend(
                (t.episode_length[ids] * t.env.step_dt).cpu().tolist()
            )
            t._finish_episodes(ids)
            t.env.reset(env_ids=ids)
            after = t.extractor.state()
            error = (
                float((after[live] - next_state[live]).abs().max())
                if bool(live.any())
                else 0.0
            )
            self.reset_live_max_error = max(self.reset_live_max_error, error)
            torch.testing.assert_close(after[live], next_state[live], rtol=0, atol=1e-6)
            torch.testing.assert_close(
                t.extractor.context()[live], live_context, rtol=0, atol=1e-7
            )
            assert torch.equal(t.command.time_steps[live], clock)
            t.last_action[ids] = 0.0
            t.episode_len[ids] = 0
            t.episode_return[ids] = 0.0
            t.episode_length[ids] = 0
            t.episode_start_clip[ids] = t.command.clip_ids[ids]
            t.episode_start_time[ids] = t.command.time_steps[ids]
            context = t.extractor.context()[ids]
            padding = t.proprio(
                after[ids],
                t.last_action[ids],
                encoder_bias_from_context(context, t.spec.num_joints),
                add_noise=True,
            )
            t.history[ids] = padding[:, None].expand(-1, t.history_len, -1)
            next_obs[ids] = self.observe(ids)
        alg.process_env_step(next_obs, learning_reward, done, {})
        return next_obs
