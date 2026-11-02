"""Local Euclidean gradient diagnostic; no claim about Adam or control safety."""
import torch


def project_extra_gradient(extra, protected):
    """Remove only the component conflicting with a protected loss gradient.

    Lists must contain finite matching tensors. No-conflict and zero-protected
    cases return original tensors unchanged. Adam moments/preconditioning and
    finite nonlinear steps are outside this local first-order guarantee.
    """
    assert len(extra) == len(protected) and extra
    assert all(g.shape == h.shape for g, h in zip(extra, protected))
    dot = sum((g.double()*h.double()).sum() for g, h in zip(extra, protected))
    norm2 = sum(h.double().square().sum() for h in protected)
    assert torch.isfinite(dot) and torch.isfinite(norm2)
    if norm2 == 0 or dot >= 0:
        return extra, dict(projected=False, dot_before=float(dot), dot_after=float(dot))
    coefficient = dot/norm2
    projected = [(g.double()-coefficient*h.double()).to(g.dtype) for g, h in zip(extra, protected)]
    after = sum((g.double()*h.double()).sum() for g, h in zip(projected, protected))
    assert all(torch.isfinite(g).all() for g in projected)
    return projected, dict(projected=True, dot_before=float(dot), dot_after=float(after))
