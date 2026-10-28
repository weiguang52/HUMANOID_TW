"""Opt-in projected Gaussian score for the actual17-action speed-gated controller.

Raw30 sampling and full30 KL remain unchanged. This module does not modify PPO,
physics, rewards, deployed actions or actor/critic observation groups by default.
"""
import torch
from candidate_action_scope import make_scope_contract
from reference_speed_blend import candidate_weight

ACTIVE = 'controlled_action_active'


def make_contract(names):
    scope = make_scope_contract(names, 'lower_body_waist')
    return {'version': 1, 'kind': 'controlled17_active_gaussian_score_entropy',
            'action_scope': scope, 'active_key': ACTIVE, 'active_dtype': 'binary_float32',
            'active_rule': 'same current reference candidate_weight > 0',
            'blend_speed_low': .05, 'blend_speed_high': .1,
            'score': 'sum raw Gaussian log_prob over selected17; zero inactive',
            'entropy': 'sum raw Gaussian entropy over selected17; zero inactive',
            'reduction': 'original all-sample mean; no active-count normalization',
            'sampling': 'unchanged raw30 stochastic sample and RNG',
            'KL': 'unchanged full30 Gaussian KL and original scheduler',
            'actor_critic_inputs': 'unchanged; auxiliary binary float32 is training-only'}


def selected_indices(contract):
    names = contract['action_scope']['action_joint_names']
    if contract != make_contract(names):
        raise ValueError('Unsupported controlled likelihood contract')
    return tuple(names.index(n) for n in contract['action_scope']['candidate_joint_names'])


def validate_resume(saved, current, *, allow_teacher_initialization=False):
    selected_indices(current)
    if saved is None:
        if not allow_teacher_initialization:
            raise ValueError('New likelihood requires exact original teacher initialization')
    elif saved != current:
        raise ValueError('Resume must preserve controlled likelihood contract')


def projected_statistics(mean, std, actions, active, indices):
    if mean.ndim != 2 or mean.shape[-1] != 30 or std.shape != mean.shape or actions.shape != mean.shape:
        raise ValueError('Expected matching batch x30 Gaussian parameters/actions')
    if active.shape != (mean.shape[0], 1) or active.dtype != torch.float32 or active.device != mean.device:
        raise ValueError('Expected same-device binary float32 batch x1 active column')
    if not bool(((active == 0) | (active == 1)).all()):
        raise ValueError('Activity must be exactly0 or1')
    indices = tuple(indices)
    if len(indices) != 17 or len(set(indices)) != 17 or any(type(i) is not int or not 0 <= i < 30 for i in indices):
        raise ValueError('Expected17 distinct selected action indices')
    index = torch.tensor(indices, dtype=torch.long, device=mean.device)
    m, s, a = (x.index_select(-1, index) for x in (mean, std, actions))
    if not bool(torch.isfinite(m).all() & torch.isfinite(s).all() & torch.isfinite(a).all() & (s > 0).all()):
        raise ValueError('Selected Gaussian parameters/actions must be finite with positive std')
    normal = torch.distributions.Normal(m, s)
    lp = normal.log_prob(a).sum(-1)
    ent = normal.entropy().sum(-1)
    return (torch.where(active[:, 0].bool(), lp, torch.zeros_like(lp)),
            torch.where(active[:, 0].bool(), ent, torch.zeros_like(ent)))


class ControlledActionActivity:
    """Add current gate to storage before runner construction; never NN inputs.

    The step guard compares the gate actually read before action selection with
    the current motion command before physics, including asynchronous resets.
    No persistent action/filter state, RNG draw or simulator step is added.
    """
    def __init__(self, env, command, trace_steps=64):
        self.env = env
        self.command = command
        self.index = command.cfg.body_names.index('base_link')
        self.current_active = None
        self.trace_steps = trace_steps
        self.trace = []
        self.completed_steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _weight(self):
        speed = torch.linalg.vector_norm(self.command.body_lin_vel_w[:, self.index, :2], dim=-1)
        return candidate_weight(speed, 1., .05, .1)

    def _pack(self, obs):
        if ACTIVE in obs:
            raise ValueError('Activity observation already present')
        p = obs['policy']
        active = (self._weight() > 0).unsqueeze(-1)
        if p.ndim != 2 or active.shape != (p.shape[0], 1) or active.device != p.device:
            raise ValueError('Activity/policy observation batch or device mismatch')
        self.current_active = active.detach().clone()
        output = obs.clone(False)
        output[ACTIVE] = self.current_active.float()
        return output

    def get_observations(self):
        return self._pack(self.env.get_observations())

    def reset(self):
        obs, extras = self.env.reset()
        return self._pack(obs), extras

    @torch.inference_mode()
    def step(self, actions):
        if self.current_active is None:
            raise RuntimeError('Get current activity observation before step')
        weight = self._weight()
        if not torch.equal(self.current_active[:, 0], weight > 0):
            raise RuntimeError('Stored current gate differs from pre-physics command')
        if len(self.trace) < self.trace_steps:
            self.trace.append({'active': self.current_active.cpu().numpy().copy(),
                               'weight': weight.cpu().numpy().copy()})
        obs, rewards, dones, extras = self.env.step(actions)
        self.completed_steps += 1
        return self._pack(obs), rewards, dones, extras

    def save_trace(self, path):
        import numpy as np
        if not self.trace:
            raise RuntimeError('No actual activity steps captured')
        np.savez_compressed(path, **{k: np.stack([r[k] for r in self.trace]) for k in self.trace[0]})
        return {'completed_steps': self.completed_steps, 'captured_steps': len(self.trace),
                'active_key': ACTIVE, 'active_dtype': 'binary_float32', 'guard': 'every pre-physics step exact gate equality'}


LEGACY_SCORE = 'raw student sample with original student log probability'
CONTROLLED_SCORE = 'raw student30 sample; active17 Gaussian score/entropy; unchanged full30 KL'


def attach_training_contract(physical, likelihood):
    selected_indices(likelihood)
    if physical.get('version') != 2 or physical.get('candidate_action_scope') != likelihood['action_scope']:
        raise ValueError('Controlled likelihood requires matching scoped physical contract')
    if physical.get('PPO_action') != LEGACY_SCORE or 'controlled_action_likelihood' in physical:
        raise ValueError('Unexpected inherited PPO likelihood metadata')
    result = dict(physical)
    result.update(version=3, PPO_action=CONTROLLED_SCORE, controlled_action_likelihood=likelihood)
    return result


def detach_training_contract(contract):
    if contract.get('version') != 3 or contract.get('PPO_action') != CONTROLLED_SCORE:
        raise ValueError('Unsupported controlled training mixture contract')
    likelihood = contract['controlled_action_likelihood']
    result = dict(contract)
    result.pop('controlled_action_likelihood')
    result.update(version=2, PPO_action=LEGACY_SCORE)
    if attach_training_contract(result, likelihood) != contract:
        raise ValueError('Malformed controlled training mixture contract')
    return result, likelihood
