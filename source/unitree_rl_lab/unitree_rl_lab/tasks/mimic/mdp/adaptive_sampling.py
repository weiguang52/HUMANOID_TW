'''Pure helpers for balanced failure-aware motion sampling.'''

from __future__ import annotations

import torch


def failure_rates(
    failed_count: torch.Tensor,
    visit_count: torch.Tensor,
    *,
    unvisited_score: float = 1.0,
) -> torch.Tensor:
    '''Return a bounded failure-rate score without raw-count feedback bias.'''
    if failed_count.shape != visit_count.shape or failed_count.ndim != 1:
        raise ValueError('failed_count and visit_count must be one-dimensional tensors with equal shape')
    if not 0.0 <= unvisited_score <= 1.0:
        raise ValueError('unvisited_score must be in [0, 1]')
    if not torch.isfinite(failed_count).all() or not torch.isfinite(visit_count).all():
        raise ValueError('failure statistics must be finite')
    if torch.any(failed_count < 0) or torch.any(visit_count < 0):
        raise ValueError('failure statistics must be non-negative')

    observed = visit_count > torch.finfo(visit_count.dtype).eps
    rates = torch.full_like(visit_count, unvisited_score)
    rates[observed] = failed_count[observed] / visit_count[observed]
    return rates.clamp_(0.0, 1.0)


def _cap_and_redistribute(
    probabilities: torch.Tensor,
    prior: torch.Tensor,
    max_probability: float,
) -> torch.Tensor:
    count = probabilities.numel()
    if count == 1:
        return torch.ones_like(probabilities)
    if not 0.0 < max_probability <= 1.0:
        raise ValueError('max_probability must be in (0, 1]')
    if max_probability * count < 1.0 - 1.0e-7:
        raise ValueError('max_probability is infeasible for the number of bins')

    result = probabilities.clone()
    for _ in range(count):
        over = result > max_probability
        if not torch.any(over):
            return result / result.sum()
        excess = (result[over] - max_probability).sum()
        result[over] = max_probability
        free = ~over
        if not torch.any(free):
            break
        weights = prior[free]
        if weights.sum() <= 0:
            weights = torch.ones_like(weights)
        result[free] += excess * weights / weights.sum()
    raise RuntimeError('failed to enforce adaptive sampling probability cap')


def balanced_sampling_probabilities(
    failure_score: torch.Tensor,
    bin_prior: torch.Tensor,
    *,
    uniform_ratio: float,
    max_probability: float | None = None,
) -> torch.Tensor:
    '''Mix normalized difficulty with an explicit bin-prior share.

    ``uniform_ratio`` is the exact mass reserved for ``bin_prior``. The old
    implementation added an unnormalized failure count to a tiny per-bin
    prior, which let already sampled failures reinforce themselves.
    '''
    if failure_score.shape != bin_prior.shape or failure_score.ndim != 1 or failure_score.numel() == 0:
        raise ValueError('failure_score and bin_prior must be non-empty one-dimensional tensors with equal shape')
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError('uniform_ratio must be in [0, 1]')
    if not torch.isfinite(failure_score).all() or not torch.isfinite(bin_prior).all():
        raise ValueError('sampling inputs must be finite')
    if torch.any(failure_score < 0) or torch.any(bin_prior < 0) or bin_prior.sum() <= 0:
        raise ValueError('sampling inputs must be non-negative and bin_prior must have positive mass')

    prior = bin_prior / bin_prior.sum()
    if failure_score.sum() <= torch.finfo(failure_score.dtype).eps:
        probabilities = prior
    else:
        difficulty = failure_score / failure_score.sum()
        probabilities = uniform_ratio * prior + (1.0 - uniform_ratio) * difficulty
    probabilities = probabilities / probabilities.sum()
    if max_probability is not None:
        probabilities = _cap_and_redistribute(probabilities, prior, max_probability)
    return probabilities
