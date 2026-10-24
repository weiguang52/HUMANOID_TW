"""Synchronize PPO's adaptive scheduler with the restored optimizer state."""
import math


def restore_learning_rate(algorithm):
    groups=algorithm.optimizer.param_groups
    rates=[float(group['lr']) for group in groups]
    if not rates or any(not math.isfinite(rate) or rate<=0 for rate in rates):
        raise ValueError('Expected positive finite optimizer learning rates')
    if any(not math.isclose(rate,rates[0],rel_tol=1e-12) for rate in rates):
        raise ValueError('Scalar PPO scheduler cannot preserve distinct group learning rates')
    previous=float(algorithm.learning_rate)
    algorithm.learning_rate=rates[0]
    return {'scheduler_before':previous,'optimizer_restored':rates,
            'scheduler_after':algorithm.learning_rate}
