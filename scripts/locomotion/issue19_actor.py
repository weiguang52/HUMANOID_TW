"""Issue19 exact equivariant Gaussian actor; no critic or environment changes."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories

class SharedGaussian(GaussianDistribution):
    def update(self, mean):
        std = self.shared_std_param.index_select(0, self.orbit_index)
        self._distribution = Normal(mean, std.expand_as(mean))

class SymmetricActor(MLPModel):
    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        x = self.get_latent(obs, None, hidden_state)
        a = self.mlp(x)
        b = self.mlp(self._issue19_mapping.actor(x))
        mean = 0.5 * (a + self._issue19_mapping.action(b))
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def as_onnx(self, verbose=False):
        return ExportActor(self)

    def as_jit(self):
        return ExportActor(self)

class ExportActor(nn.Module):
    """A single graph, two shared-weight forwards, 101 inputs and 30 outputs."""
    def __init__(self, actor):
        super().__init__()
        self.mlp = copy.deepcopy(actor.mlp)
        m = actor._issue19_mapping
        self.register_buffer('joint_perm', m.spatial.perm.clone())
        self.register_buffer('joint_sign', m.spatial.sign.clone())
        self.register_buffer('action_perm', m.spatial.action_perm.clone())
        self.register_buffer('action_sign', m.spatial.action_factor.clone())

    def forward(self, x):
        y = torch.cat((
            x[..., :3] * x.new_tensor([-1., 1., -1.]),
            x[..., 3:6] * x.new_tensor([1., -1., 1.]),
            x[..., 6:9] * x.new_tensor([1., -1., -1.]),
            x[..., 9:39].index_select(-1, self.joint_perm) * self.joint_sign,
            x[..., 39:69].index_select(-1, self.joint_perm) * self.joint_sign,
            x[..., 69:99].index_select(-1, self.joint_perm) * self.joint_sign,
            -x[..., 99:101]), dim=-1)
        return 0.5 * (self.mlp(x) + self.mlp(y).index_select(-1, self.action_perm) * self.action_sign)

    def get_dummy_inputs(self):
        return (torch.zeros(1, 101),)

    @property
    def input_names(self):
        return ['obs']

    @property
    def output_names(self):
        return ['actions']

def install_actor(actor, mapping, optimizer=None):
    """Install after loading the complete parent, without drawing RNG values."""
    assert type(actor) is MLPModel and not actor.obs_normalization
    assert actor.obs_groups == ['policy'] and actor.obs_dim == 101
    d = actor.distribution
    assert type(d) is GaussianDistribution and d.std_type == 'scalar'
    old = d.std_param
    before = old.detach().clone()
    assert (before > 0).all()
    perm = mapping.spatial.action_perm
    groups = []
    orbit = torch.empty(30, dtype=torch.long, device=old.device)
    for i in range(30):
        if i > int(perm[i]):
            continue
        members = sorted(set([i, int(perm[i])]))
        orbit[members] = len(groups)
        groups.append(members)
    def reduce(value):
        return torch.stack([value[g].mean() for g in groups])
    new = nn.Parameter(reduce(before))
    saved_state = {}
    if optimizer is not None:
        found = 0
        for group in optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p is old:
                    group['params'][i] = new
                    found += 1
        assert found == 1
        original_state = optimizer.state.pop(old, {})
        for key, value in original_state.items():
            saved_state[key] = reduce(value) if isinstance(value, torch.Tensor) and value.shape == old.shape else copy.deepcopy(value)
        optimizer.state[new] = saved_state
    del d._parameters['std_param']
    d.register_parameter('shared_std_param', new)
    d.register_buffer('orbit_index', orbit)
    d.__class__ = SharedGaussian
    actor._issue19_mapping = mapping
    actor.__class__ = SymmetricActor
    after = new.detach().index_select(0, orbit)
    assert torch.equal(after, after[perm])
    return dict(enabled=True, parent_std=before.cpu().tolist(),
                shared_std=after.cpu().tolist(), orbit_members=groups,
                std_parameters_before=30, std_parameters_after=len(groups),
                optimizer_std_moments='orbit arithmetic mean; scalar step retained',
                optimizer_std_state_keys=list(saved_state),
                all_other_parameters_and_optimizer_states_unchanged=True)

def install_runner(runner, base, mode, output):
    assert mode in ['off', 'sym_wrap', 'sym_cont']
    if mode == 'off':
        return
    from issue17_symmetry import mapping_for_env
    mapping = mapping_for_env(base)
    report = dict(enabled=False, true_noop=True)
    if mode == 'sym_wrap':
        report = install_actor(runner.alg.actor, mapping, runner.alg.optimizer)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'actor_structure_audit.json').write_text(json.dumps(report, indent=2) + '\n')
