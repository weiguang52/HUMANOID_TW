"""Mass-metric joint-speed projection with floating-base momentum conservation."""

import math
import torch


@torch.no_grad()
def project_velocity_mass_metric(velocity, mass, caps, free_dofs=6, tolerance=1e-5):
    """Solve the box-constrained velocity projection in the physical inertia metric.

    Eliminate unconstrained root velocities, solve the hinge box QP with FISTA,
    then recover the root impulse reaction. Unlike component clipping, this
    preserves free generalized momentum and cannot add significant kinetic energy.
    """
    hinge = velocity[:, free_dofs:]
    bounds = caps[free_dofs:]
    if not bool((hinge.abs() > bounds).any()):
        return velocity, dict(
            active=False,
            iterations=0,
            residual=0.0,
            momentum_error=0.0,
            energy_increase=0.0,
        )
    # Policy TF32 must not reduce precision of physical constraint impulses.
    original_dtype = velocity.dtype
    velocity, mass, caps = velocity.double(), mass.double(), caps.double()
    hinge, bounds = velocity[:, free_dofs:], caps[free_dofs:]
    ff = mass[:, :free_dofs, :free_dofs]
    fh = mass[:, :free_dofs, free_dofs:]
    response = torch.linalg.solve(ff, fh)
    reduced = (
        mass[:, free_dofs:, free_dofs:] - mass[:, free_dofs:, :free_dofs] @ response
    )
    reduced = 0.5 * (reduced + reduced.transpose(-1, -2))
    lipschitz = reduced.abs().sum(-1).amax(-1).clamp_min(torch.finfo(mass.dtype).eps)
    scaled = reduced / lipschitz[:, None, None]
    x = hinge.clamp(-bounds, bounds)
    y = x.clone()
    acceleration = 1.0
    residual = float("inf")
    iterations = 0
    for block in range(4):
        for _ in range(64):
            gradient = (scaled @ (y - hinge).unsqueeze(-1)).squeeze(-1)
            candidate = (y - gradient).clamp(-bounds, bounds)
            next_acceleration = (1 + math.sqrt(1 + 4 * acceleration**2)) / 2
            y = candidate + (acceleration - 1) / next_acceleration * (candidate - x)
            x = candidate
            acceleration = next_acceleration
            iterations += 1
        gradient = (scaled @ (x - hinge).unsqueeze(-1)).squeeze(-1)
        residual = float((x - (x - gradient).clamp(-bounds, bounds)).abs().max())
        if residual <= tolerance:
            break
    if residual > tolerance:
        raise RuntimeError(f"Velocity QP did not converge: residual={residual}")
    root = velocity[:, :free_dofs] - (response @ (x - hinge).unsqueeze(-1)).squeeze(-1)
    projected = torch.cat([root, x], dim=-1).to(original_dtype)
    before = (mass @ velocity.unsqueeze(-1)).squeeze(-1)
    after = (mass @ projected.to(mass.dtype).unsqueeze(-1)).squeeze(-1)
    momentum_error = float((after[:, :free_dofs] - before[:, :free_dofs]).abs().max())
    energy_before = 0.5 * (velocity * before).sum(-1)
    energy_after = 0.5 * (projected * after).sum(-1)
    increase = float((energy_after - energy_before).clamp_min(0).max())
    if not bool(torch.isfinite(projected).all()):
        raise FloatingPointError("Non-finite projected velocity")
    if momentum_error > 1e-5 * (1 + float(before[:, :free_dofs].abs().max())):
        raise RuntimeError(
            f"Projection changed free generalized momentum: {momentum_error}"
        )
    if bool((energy_after > energy_before + 1e-6 * (1 + energy_before.abs())).any()):
        raise RuntimeError(f"Projection injected kinetic energy: {increase}")
    return projected, dict(
        active=True,
        iterations=iterations,
        residual=residual,
        momentum_error=momentum_error,
        energy_increase=increase,
    )
