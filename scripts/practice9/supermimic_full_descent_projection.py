"""Small Gram-space projection for frozen fit diagnostics only."""
import torch


def project_full_descent(gram, common_weights, cycles=512):
    gram = gram.detach().double().cpu()
    weights = common_weights.detach().double().cpu()
    assert gram.shape == (len(weights), len(weights))
    assert torch.isfinite(gram).all() and torch.isfinite(weights).all()
    assert torch.allclose(gram, gram.T, atol=1e-12, rtol=1e-12)
    target = torch.zeros_like(weights)
    target[1] = -1.
    bound = -gram @ weights
    rhs = gram @ target - bound
    dual = torch.zeros_like(weights)
    for _ in range(cycles):
        for i in range(len(weights)):
            if float(gram[i, i]) > 1e-30:
                dual[i] = max(0., float(dual[i] + (rhs[i] - gram[i] @ dual) / gram[i, i]))
    coefficients = target - dual
    residual = gram @ coefficients - bound
    distance = lambda c: float((c-target) @ gram @ (c-target))
    fallback = bool(residual.max() > 1e-10 or distance(coefficients) > distance(-weights) + 1e-10)
    if fallback:
        coefficients = -weights
    return coefficients, {
        'cycles': cycles, 'fallback_common': fallback,
        'pre_fallback_max_constraint_residual': float(residual.max()),
        'max_constraint_residual': float((gram @ coefficients-bound).max()),
        'dual_complementarity_residual': float((dual * residual).abs().max()),
        'common_slopes': bound.tolist(),
        'projected_slopes': (gram @ coefficients).tolist(),
        'target_distance_squared': distance(coefficients),
        'common_target_distance_squared': distance(-weights),
        'scope': 'Unnormalized unit-gradient space; finite-step nine-loss checks still required',
    }
