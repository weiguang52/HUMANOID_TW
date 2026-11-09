"""Opt-in issue #6 mass bounds and independent startup audit."""
import torch
from isaaclab.envs.mdp import events

MASS_SCALE_RANGE = (0.5, 1.5)

def configure_mass_randomization(env_cfg, profile):
    if profile == "legacy":
        return
    if profile != "bounded":
        raise ValueError(profile)
    term = env_cfg.events.add_base_mass
    if term is not None:
        # Same base_link selection, friction, inertia recomputation and startup
        # timing as robust08. Scale nominal mass rather than adding kilograms.
        term.params["operation"] = "scale"
        term.params["distribution"] = "uniform"
        term.params["mass_distribution_params"] = MASS_SCALE_RANGE

class AuditedMassRandomization(events.randomize_rigid_body_mass):
    """Capture requested masses immediately before Isaac Lab's safety clamp."""
    def __call__(self, env, env_ids, asset_cfg, mass_distribution_params, operation,
                 distribution="uniform", recompute_inertia=True, min_mass=1.e-6):
        original = events._randomize_prop_by_op
        def capture(*args, **kw):
            value = original(*args, **kw)
            env._issue6_requested_masses = value.detach().clone()
            return value
        events._randomize_prop_by_op = capture
        try:
            super().__call__(env, env_ids, asset_cfg, mass_distribution_params, operation,
                             distribution, recompute_inertia, min_mass)
        finally:
            events._randomize_prop_by_op = original

def mass_distribution_report(robot, requested=None):
    applied = robot.root_physx_view.get_masses().cpu()
    nominal = robot.data.default_mass.cpu()
    assert applied.shape == nominal.shape
    assert torch.isfinite(applied).all() and (applied > 0).all()
    report = {"num_envs": applied.shape[0], "bodies": [], "clamp_kg": 1.e-6}
    for i, name in enumerate(robot.body_names):
        a=applied[:,i];n=nominal[:,i];ratio=a/n
        raw=None if requested is None else requested[:,i].cpu()
        report["bodies"].append({
            "body":name,"nominal_min_kg":n.min().item(),"nominal_max_kg":n.max().item(),
            "applied_min_kg":a.min().item(),"applied_max_kg":a.max().item(),
            "applied_mean_kg":a.mean().item(),
            "applied_quantiles_kg":torch.quantile(a,torch.tensor([0.,.01,.05,.5,.95,.99,1.])).tolist(),
            "minimum_nominal_ratio":ratio.min().item(),
            "below_1pct_nominal_count":int((ratio < .01).sum()),
            "below_1pct_nominal_fraction":(ratio < .01).float().mean().item(),
            "at_safety_clamp_count":int(torch.isclose(a,torch.tensor(1.e-6),atol=1.e-10,rtol=0).sum()),
            "requested_at_or_below_clamp_count":None if raw is None else int((raw <= 1.e-6).sum()),
            "requested_min_kg":None if raw is None else raw.min().item(),
        })
    return report
