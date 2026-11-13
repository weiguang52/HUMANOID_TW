"""Issue18 inference projection. No learning, parameter edits, or clock changes."""
import json
from pathlib import Path
from issue17_symmetry import mapping_for_env
from issue16_mirror import install_policy as validate_runtime_contract

class StrictSymmetricPolicy:
    def __init__(self, policy, mapping, mode='strict', base=None):
        assert mode in ('original', 'identity', 'strict')
        self.policy, self.mapping, self.mode = policy, mapping, mode
        self.calls = 0
        self.base = base
        self.last_output = None
        self.pre_action_episode_step = None
        self.feedback_checks = 0
        self.feedback_max_error = 0.0

    def __call__(self, observations):
        self.calls += 1
        if self.base is not None:
            import torch
            last = self.base.action_manager.action
            error = float((observations["policy"][...,69:99]-last).abs().max())
            self.feedback_max_error = max(self.feedback_max_error, error)
            assert torch.equal(observations["policy"][...,69:99], last), "last_action does not match actual action-manager feedback"
            self.feedback_checks += 1
            self.pre_action_episode_step = self.base.episode_length_buf.detach().clone()
        original = self.policy(observations)
        if self.mode in ('original', 'identity'):
            self.last_output = original.detach().clone()
            return original
        reflected = observations.clone() if hasattr(observations, 'clone') else dict(observations)
        reflected['policy'] = self.mapping.actor(observations['policy'])
        mirrored_action = self.mapping.action(self.policy(reflected))
        output = 0.5 * (original + mirrored_action)
        self.last_output = output.detach().clone()
        return output


def install_policy(policy, base, mode, output):
    # Reuse the actual-axis/default-offset/normalizer checks, but keep all new
    # provenance in the caller's issue18 output directory.
    _, metadata = validate_runtime_contract(policy, base, 'original', output)
    mapping = mapping_for_env(base)
    metadata.update(mode=mode, intervention='strict group-average inference projection',
                    gait_reflection='sin,cos both negated: training half-cycle transform',
                    formula='0.5*(pi(s)+Ma*pi(Ts))', training_updates=0,
                    configuration_changed=False, original_clock_preserved=True)
    Path(output, 'mirror_runtime_contract.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return StrictSymmetricPolicy(policy, mapping, mode, base=base), metadata
