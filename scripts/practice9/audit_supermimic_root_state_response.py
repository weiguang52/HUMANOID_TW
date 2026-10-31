"""Compare frozen model and native root-velocity transition sensitivity."""

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
plan = json.loads((a.collection / "plan.json").read_text())
integrity = json.loads((a.collection / "integrity.json").read_text())
assert integrity["passed"] and plan["contract"] == "root_velocity_state_branches_v1"
hashes = {x["file"]: x["sha256"] for x in integrity["files"]}
rows = json.loads((a.collection / "results.json").read_text())["rows"]
loaded = {}
for r in rows:
    path = (a.collection / r["file"]).resolve()
    assert (
        path.is_relative_to(a.collection.resolve())
        and hashlib.sha256(path.read_bytes()).hexdigest() == hashes[r["file"]]
    )
    with np.load(path) as z:
        d = {k: z[k].copy() for k in ["state", "action", "context", "next_state"]}
    loaded[
        (r["split"], r["warmup"], r["kind"], r.get("level"), r.get("assignment"))
    ] = d
scale = np.array(plan["base_delta"], dtype=float)


def derivative(a, b, level):
    return (a[1::2] - a[::2] + b[::2] - b[1::2]) / (4 * level * scale[:, None])


def metrics(actual, pred):
    error = pred - actual
    x = actual.reshape(-1)
    y = pred.reshape(-1)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "relative_error_norm": float(np.linalg.norm(error) / np.linalg.norm(actual))
        if np.linalg.norm(actual) > 0
        else None,
        "cosine": float(x @ y / den) if den > 0 else None,
    }


def block_stats(matrix):
    block = matrix[:, 7:13].T
    normalized = block * scale[None, :] / scale[:, None]
    return {
        "root_velocity_jacobian_output_by_input": block.tolist(),
        "spectral_radius": float(np.abs(np.linalg.eigvals(block)).max()),
        "scaled_singular_max": float(np.linalg.svd(normalized, compute_uv=False).max()),
    }


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
    conditions = []
    with torch.no_grad():
        for split in ["fit", "validation"]:
            for warmup in plan[split + "_warmups"]:
                for level in plan["levels"]:
                    ds = [loaded[(split, warmup, "branch", level, i)] for i in [0, 1]]
                    pred = []
                    for d in ds:
                        args = [
                            torch.as_tensor(d[k], device="cuda:0")
                            for k in ["state", "action", "context"]
                        ]
                        v = model(*args)
                        assert torch.isfinite(v).all()
                        pred.append(v.cpu().numpy().astype(float))
                    actual = derivative(
                        *[d["next_state"].astype(float) for d in ds], level
                    )
                    predicted = derivative(*pred, level)
                    conditions.append(
                        {
                            "split": split,
                            "warmup": warmup,
                            "level": level,
                            "actual": block_stats(actual),
                            "predicted": block_stats(predicted),
                            "normalized_input_root_linear_response": metrics(
                                actual[:, 7:10] * scale[:, None],
                                predicted[:, 7:10] * scale[:, None],
                            ),
                            "normalized_input_root_angular_response": metrics(
                                actual[:, 10:13] * scale[:, None],
                                predicted[:, 10:13] * scale[:, None],
                            ),
                            "normalized_input_root_position_response": metrics(
                                actual[:, :3] * scale[:, None],
                                predicted[:, :3] * scale[:, None],
                            ),
                        }
                    )
        for k, v in base.state_dict().items():
            assert torch.equal(v.cpu(), w[k]), k
    reports.append(
        {
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "conditions": conditions,
            "weights_and_norms_unchanged": True,
        }
    )
report = {
    "scope": "All declared native state-response conditions, no fitting or policy promotion. Root-velocity6x6 subblock eigenvalues are descriptive, not full-state or closed-loop stability proofs. Singular gains use the declared linear/angular perturbation scales. Response errors normalize input axes by declared perturbation scales; output groups retain their physical units. No acceptance gate introduced.",
    "collection_plan_sha256": hashlib.sha256(
        (a.collection / "plan.json").read_bytes()
    ).hexdigest(),
    "source_hashes": hashes,
    "models": reports,
}
(a.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print("NATIVE_STATE_RESPONSE_AUDITED", len(reports))
