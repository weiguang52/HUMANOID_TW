"""Oracle-only feedback ablations to locate rollout error amplification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from supermimic.algo.networks import WorldModel, WorldModelCfg
from supermimic_pose_integration_diagnostic import apply_offline_prediction_contract

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--collection", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, action="append", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir()
modes = {
    "baseline": None,
    "root_position": slice(0, 3),
    "root_orientation": slice(3, 7),
    "root_linear_velocity": slice(7, 10),
    "root_angular_velocity": slice(10, 13),
    "joint_position": slice(13, 43),
    "joint_velocity": slice(43, 73),
    "all_state": slice(0, 73),
}
plan = {
    "scope": "Oracle feedback diagnostic only. Uses previous-frame truth after the initial step; never a deployable predictor or control acceptance. No fitting or policy promotion.",
    "modes": list(modes),
    "horizons": [8, 32],
    "selection": "All contiguous H32 windows of every existing temporal-validation trajectory, no replacement or reset crossing.",
}
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
arrays = {k: [] for k in ["state", "action", "context", "next_state"]}
sources = []
trace_ids = []
for row in json.loads((a.collection / "results.json").read_text())["rows"]:
    if row["split"] != "validation":
        continue
    path = (a.collection / row["trace"]).resolve()
    assert path.is_relative_to(a.collection.resolve())
    with np.load(path) as z:
        d = {k: z[k].copy() for k in arrays}
    n = len(d["state"])
    assert np.array_equal(d["next_state"][:-1], d["state"][1:])
    for k, width in [
        ("state", 73),
        ("action", 30),
        ("context", 34),
        ("next_state", 73),
    ]:
        assert d[k].shape == (n, width) and np.isfinite(d[k]).all()
    count = max(0, n - 31)
    sources.append(
        {
            "trace": row["trace"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "windows": count,
        }
    )
    for start in range(count):
        arrays["state"].append(d["state"][start])
        for k in ["action", "context", "next_state"]:
            arrays[k].append(d[k][start : start + 32])
        trace_ids.append(len(sources) - 1)
assert len(sources) == 8 and arrays["state"]
d = {k: torch.as_tensor(np.stack(v), device="cuda:0") for k, v in arrays.items()}
ids = np.array(trace_ids)
reports = []
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
    first = None
    with torch.no_grad():
        for mode, sl in modes.items():
            pred = d["state"].clone()
            for step in range(32):
                if step > 0 and sl is not None:
                    pred = pred.clone()
                    pred[:, sl] = d["next_state"][:, step - 1, sl]
                pred = model(pred, d["action"][:, step], d["context"][:, step])
                assert torch.isfinite(pred).all()
                if step == 0:
                    if first is None:
                        first = pred.clone()
                    else:
                        assert torch.equal(first, pred), "Oracle affected first step"
                if step + 1 not in [8, 32]:
                    continue
                truth = d["next_state"][:, step]
                pos = (
                    torch.linalg.vector_norm(pred[:, :3] - truth[:, :3], dim=1)
                    .cpu()
                    .numpy()
                )
                q = torch.nn.functional.normalize(pred[:, 3:7], dim=1)
                t = torch.nn.functional.normalize(truth[:, 3:7], dim=1)
                t = torch.where((q * t).sum(1, keepdim=True) < 0, -t, t)
                angle = (
                    (
                        4
                        * torch.asin(
                            (torch.linalg.vector_norm(q - t, dim=1) / 2).clamp(0, 1)
                        )
                    )
                    .cpu()
                    .numpy()
                )
                pertrace = []
                for i, src in enumerate(sources):
                    mask = ids == i
                    if mask.any():
                        pertrace.append(
                            {
                                "trace": src["trace"],
                                "windows": int(mask.sum()),
                                "root_position_p95_m": float(
                                    np.quantile(pos[mask], 0.95)
                                ),
                                "root_orientation_p95_rad": float(
                                    np.quantile(angle[mask], 0.95)
                                ),
                            }
                        )
                row = {
                    "mode": mode,
                    "horizon": step + 1,
                    "windows": len(pred),
                    "median_trace_root_position_p95_m": float(
                        np.median([x["root_position_p95_m"] for x in pertrace])
                    ),
                    "median_trace_root_orientation_p95_rad": float(
                        np.median([x["root_orientation_p95_rad"] for x in pertrace])
                    ),
                    "per_trace": pertrace,
                }
                rows.append(row)
                if step == 31:
                    print(
                        path.parent.parent.parent.name,
                        mode,
                        row["median_trace_root_position_p95_m"],
                        row["median_trace_root_orientation_p95_rad"],
                        flush=True,
                    )
        for k, v in base.state_dict().items():
            assert torch.equal(v.cpu(), w[k]), k
    reports.append(
        {
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first_step_identical_all_modes": True,
            "weights_and_norms_unchanged": True,
            "rows": rows,
        }
    )
(a.output / "audit.json").write_text(
    json.dumps(
        {"plan": plan, "sources": sources, "models": reports}, indent=2, allow_nan=False
    )
)
print("FEEDBACK_DIAGNOSTIC_COMPLETE", len(reports), len(ids))
