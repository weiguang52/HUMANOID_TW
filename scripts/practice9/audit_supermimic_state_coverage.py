"""Read-only fit/temporal-validation coverage and residual-bias diagnosis."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from supermimic.algo.networks import WorldModel, WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, action="append", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir()
pools = {}
sources = []
for source, folder in [
    ("trajectory", "static_perturb_v1"),
    ("branch", "orthogonal_branches_v1"),
]:
    root = a.root / folder
    for row in json.loads((root / "results.json").read_text())["rows"]:
        split = row["split"]
        assert split in ["fit", "validation"]
        path = (
            root / (row["trace"] if source == "trajectory" else row["file"])
        ).resolve()
        assert path.is_relative_to(root.resolve())
        with np.load(path) as z:
            d = {k: z[k].copy() for k in ["state", "action", "context", "next_state"]}
        n = len(d["state"])
        for k, width in [
            ("state", 73),
            ("action", 30),
            ("context", 34),
            ("next_state", 73),
        ]:
            assert d[k].shape == (n, width) and np.isfinite(d[k]).all()
        pools.setdefault((source, split), []).append(d)
        sources.append(
            {
                "source": source,
                "split": split,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
pools = {
    key: {
        k: torch.as_tensor(np.concatenate([x[k] for x in ds]), device="cuda:0")
        for k in ds[0]
    }
    for key, ds in pools.items()
}
models = []
for path in a.checkpoint:
    c = torch.load(path, map_location="cpu", weights_only=False)
    w = c["world_model"]
    cfg = c["cfg"]
    base = WorldModel(
        WorldModelCfg(
            state_dim=73,
            action_dim=30,
            context_dim=34,
            goal_dim=w["goal_norm.mean"].numel(),
            hidden_dims=tuple(cfg["wm_hidden_dims"]),
            termination_hidden=tuple(cfg["wm_term_hidden_dims"]),
        )
    ).cuda()
    base.load_state_dict(w)
    base.eval()
    model = apply_offline_prediction_contract(base, c)
    rows = []
    with torch.no_grad():

        def inputs(d, base=base):
            return torch.cat(
                [
                    base.state_norm.normalize(d["state"]),
                    base.action_norm.normalize(d["action"]),
                    base.context_norm.normalize(d["context"]),
                ],
                dim=1,
            )

        for source in ["trajectory", "branch"]:
            fit = inputs(pools[(source, "fit")])
            val = inputs(pools[(source, "validation")])
            distances = []
            for start in range(0, len(val), 128):
                distances.append(
                    torch.cdist(
                        val[start : start + 128],
                        fit,
                        compute_mode="donot_use_mm_for_euclid_dist",
                    )
                    .min(dim=1)
                    .values
                    / np.sqrt(fit.shape[1])
                )
            nearest = torch.cat(distances)
            outside = (
                (
                    (val < fit.min(dim=0).values - 1e-6)
                    | (val > fit.max(dim=0).values + 1e-6)
                )
                .float()
                .mean(dim=0)
            )
            coverage = {
                "nearest_fit_normalized_input_rms_p50": float(nearest.median()),
                "nearest_fit_normalized_input_rms_p95": float(nearest.quantile(0.95)),
                "fraction_outside_fit_range_by_feature": outside.cpu().tolist(),
            }
            for split in ["fit", "validation"]:
                d = pools[(source, split)]
                pred = []
                for start in range(0, len(d["state"]), 256):
                    pred.append(
                        model(
                            d["state"][start : start + 256],
                            d["action"][start : start + 256],
                            d["context"][start : start + 256],
                        )
                    )
                pred = torch.cat(pred)
                assert torch.isfinite(pred).all()
                error = pred - d["next_state"]
                metrics = {}
                for name, sl in [
                    ("root_linear_velocity", slice(7, 10)),
                    ("root_angular_velocity", slice(10, 13)),
                    ("joint_velocity", slice(43, 73)),
                ]:
                    e = error[:, sl]
                    mean = e.mean(dim=0)
                    mse = e.square().mean()
                    centered = (e - mean).square().mean()
                    assert torch.allclose(
                        mse, centered + mean.square().mean(), atol=1e-8, rtol=1e-5
                    )
                    metrics[name] = {
                        "mean_error": mean.cpu().tolist(),
                        "rmse": float(mse.sqrt()),
                        "centered_rmse": float(centered.sqrt()),
                        "mean_bias_fraction_of_mse": float(mean.square().mean() / mse)
                        if mse > 0
                        else 0.0,
                    }
                rows.append(
                    {
                        "source": source,
                        "split": split,
                        "frames": len(pred),
                        "metrics": metrics,
                        "coverage": coverage if split == "validation" else None,
                    }
                )
        for k, v in base.state_dict().items():
            assert torch.equal(v.cpu(), w[k]), k
    models.append(
        {
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows,
        }
    )
report = {
    "scope": "Temporal static-state holdouts only; descriptive coverage and common-bias fractions are not acceptance gates. No fitting, weight selection, or policy promotion.",
    "sources": sources,
    "models": models,
    "weights_and_normalizers_unchanged": True,
}
(a.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print("COVERAGE_AUDITED", len(models))
