"""Observe a small real-PPO run without changing rewards, draws or termination gates."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from supermimic_real_ppo import RealFeedbackCollector
import train_supermimic_real_ppo as training

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evidence', type=Path, required=True)
a, rest = p.parse_known_args()
assert not a.evidence.exists(), 'Never overwrite diagnostic evidence'
a.evidence.parent.mkdir(parents=True, exist_ok=True)
records = []
normalized = {}
original_step = ManagerBasedRlEnv.step
original_observe = RealFeedbackCollector.observe


def stats(name, value):
    value = value.detach().float().abs().reshape(-1).cpu()
    normalized.setdefault(name, []).append(value)


@torch.no_grad()
def observe(self, ids=None):
    result = original_observe(self, ids)
    t = self.t
    actor = result['actor']
    history = actor[:, :t.actor_critic.cfg.obs_dim].reshape(len(actor), t.history_len, -1)
    goal = actor[:, t.actor_critic.cfg.obs_dim:]
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    for name, value, norm in [('proprio', history, t.actor_critic.obs_norm), ('goal', goal, t.actor_critic.goal_norm), ('critic', result['critic'], t.actor_critic.critic_norm)]:
        stats(name + '_raw_abs', value)
        stats(name + '_normalized_abs', norm(value))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    return result


@torch.no_grad()
def step(self, action):
    command = self.command_manager.get_term('motion')
    clips = command.clip_ids.detach().cpu().tolist()
    frames = command.time_steps.detach().cpu().tolist()
    output = original_step(self, action)
    terminated, truncated = output[2], output[3]
    manager = self.termination_manager
    terms = {name: manager.get_term(name).detach().cpu().tolist() for name in manager.active_terms}
    for i, (clip, frame) in enumerate(zip(clips, frames)):
        records.append({'env':i, 'clip':clip, 'frame':frame, 'terminated':bool(terminated[i]), 'truncated':bool(truncated[i]), 'reasons':[name for name, flags in terms.items() if flags[i]]})
    return output


RealFeedbackCollector.observe = observe
ManagerBasedRlEnv.step = step
sys.argv = [sys.argv[0]] + rest
status = 'failed_preserve_evidence'
try:
    training.main()
    status = 'completed_observation_not_control_acceptance'
finally:
    RealFeedbackCollector.observe = original_observe
    ManagerBasedRlEnv.step = original_step
    report = {}
    for name, values in normalized.items():
        v = torch.cat(values)
        report[name] = {'count':v.numel(), 'finite':bool(torch.isfinite(v).all()), 'p50':float(v.quantile(.5)), 'p95':float(v.quantile(.95)), 'p99':float(v.quantile(.99)), 'max':float(v.max()), 'fraction_over_10':float((v>10).float().mean())}
    counts = Counter(x['clip'] for x in records)
    reason_counts = Counter(name for x in records for name in x['reasons'])
    a.evidence.write_text(json.dumps({'status':status, 'scope':'Instrumented tiny run; observation samples include reset padding and are not independent. Synchronization may alter GPU numerical replay. No RNG draws added. Not walking acceptance.', 'transition_count':len(records), 'sampled_clip_transition_counts':dict(counts), 'termination_term_counts':dict(reason_counts), 'terminated_transitions':sum(x['terminated'] for x in records), 'truncated_transitions':sum(x['truncated'] for x in records), 'normalization':report, 'transitions':records}, indent=2)+'\n')
