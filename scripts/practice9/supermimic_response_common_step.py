"""Fit-only five-loss finite-step experiment; never a policy acceptance gate."""
import copy
import math
import torch

FACTORS = (1., .5, .25, .125, .0625, .03125, .015625, .0078125)

def dot(xs, ys):
    return float(sum((x.double() * y.double()).sum() for x, y in zip(xs, ys)))

def response_common_step(parameters, optimizer, gradients, evaluate, baseline, clip, enabled):
    parameters = list(parameters)
    assert type(enabled) is bool and len(gradients) == len(baseline) == 5
    assert all(len(g) == len(parameters) for g in gradients)
    assert all(math.isfinite(v) for v in baseline)
    assert all(torch.isfinite(v).all() for g in gradients for v in g)
    before = [p.detach().clone() for p in parameters]
    old = copy.deepcopy(optimizer.state_dict())
    group = old['param_groups'][0]
    assert len(old['param_groups']) == 1 and group['weight_decay'] == 0 and not group['amsgrad']
    def apply(delta, factor):
        with torch.no_grad():
            for p, start, d in zip(parameters, before, delta):
                p.copy_(start if factor == 0 else start + factor * d)
    try:
        shadow = [torch.nn.Parameter(p.clone()) for p in before]
        shadow_opt = torch.optim.Adam(shadow, lr=group['lr'])
        shadow_opt.load_state_dict(copy.deepcopy(old))
        for p, gradient in zip(shadow, gradients[1]):
            p.grad = gradient.detach().clone()
        norm = torch.nn.utils.clip_grad_norm_(shadow, clip)
        assert torch.isfinite(norm)
        shadow_opt.step()
        adam_delta = [p.detach() - start for p, start in zip(shadow, before)]
        norms = [dot(g, g) ** .5 for g in gradients]
        # A zero objective gradient gives no strict common descent certificate.
        if any(n == 0 for n in norms):
            weights = torch.full((5,), .2, dtype=torch.float64)
            slopes = [0.] * 5
            delta = [torch.zeros_like(p) for p in before]
        else:
            units = [[v / n for v in g] for g, n in zip(gradients, norms)]
            gram = torch.tensor([[dot(x, y) for y in units] for x in units], dtype=torch.float64)
            weights = torch.full((5,), .2, dtype=torch.float64)
            for _ in range(256):
                vertex = torch.zeros_like(weights)
                vertex[torch.argmin(gram @ weights)] = 1
                difference = vertex - weights
                denominator = float(difference @ gram @ difference)
                if denominator <= 1e-30:
                    break
                alpha = max(0., min(1., -float(difference @ gram @ weights) / denominator))
                weights += alpha * difference
            direction = [-sum(float(w) * unit[i] for w, unit in zip(weights, units)) for i in range(len(parameters))]
            squared = dot(direction, direction)
            scale = (dot(adam_delta, adam_delta) / squared) ** .5 if squared > 0 else 0.
            delta = [(start + v * scale) - start for start, v in zip(before, direction)]
            slopes = (-gram @ weights).tolist()
        probes = []
        selected = None
        for factor in FACTORS:
            apply(delta, factor)
            with torch.no_grad():
                values = tuple(float(v) for v in evaluate())
            assert len(values) == 5
            accepted = all(math.isfinite(v) and v <= b for v, b in zip(values, baseline))
            probes.append({'factor': factor, 'values': [v if math.isfinite(v) else None for v in values], 'accepted': accepted})
            if accepted and selected is None:
                selected = (factor, values)
        apply(delta, 0.)
        with torch.no_grad():
            identity = tuple(float(v) for v in evaluate())
        assert len(identity) == 5 and all(abs(v-b) <= 1e-6 * max(1., abs(b)) for v,b in zip(identity, baseline)), 'Identity replay inconsistent'
        if enabled:
            factor, values = selected if selected is not None else (0., identity)
            apply(delta, factor)
        else:
            factor, values = 1., None
            with torch.no_grad():
                for p, s in zip(parameters, shadow):
                    p.copy_(s)
        optimizer.load_state_dict(shadow_opt.state_dict())
        return {'enabled': enabled, 'baseline': list(baseline), 'weights': weights.tolist(),
                'normalized_slopes': slopes, 'probes': probes, 'identity': list(identity),
                'selected_factor': factor, 'applied_values': values,
                'moments_contract': 'Full-gradient shadow Adam state for both arms, including identity',
                'scope': 'Same fit batch only; no validation or control guarantee'}
    except BaseException:
        apply([torch.zeros_like(p) for p in before], 0.)
        optimizer.load_state_dict(old)
        raise
