"""Choose one response coefficient from frozen-parent fit losses only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from supermimic.algo.networks import WorldModel, WorldModelCfg
from supermimic_antithetic_response import AntitheticResponseLoss
from supermimic_pose_integration_diagnostic import CONTRACT, PoseVelocityModel

p = argparse.ArgumentParser(description=__doc__)
for k in ["root", "parent", "integration-audit", "pair-audit", "output"]:
    p.add_argument("--" + k, type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir()
plan = {
    "rule": "coefficient = mean existing H8/H1 base loss / mean response loss on the unmodified parent, all fit samples. One coefficient, no search or validation selection.",
    "parent_sha256": hashlib.sha256(a.parent.read_bytes()).hexdigest(),
    "frozen": True,
    "policy_artifact": False,
}
(a.output / "plan.json").write_text(json.dumps(plan, indent=2))
c = torch.load(a.parent, map_location="cpu", weights_only=False)
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
meta = json.loads(a.integration_audit.read_text())
assert meta["fit_frames"] == 4505
model = PoseVelocityModel(
    base,
    {
        "contract": CONTRACT,
        "alpha": meta["alpha"],
        "dt": meta["plan"]["dt"],
        "fit_audit_sha256": hashlib.sha256(
            a.integration_audit.read_bytes()
        ).hexdigest(),
    },
)
sequences = []
h1 = []
files = []
for source, folder in [
    ("trajectory", "static_perturb_v1"),
    ("branch", "orthogonal_branches_v1"),
]:
    root = a.root / folder
    for row in json.loads((root / "results.json").read_text())["rows"]:
        if row["split"] != "fit":
            continue
        path = (
            root / (row["trace"] if source == "trajectory" else row["file"])
        ).resolve()
        assert path.is_relative_to(root.resolve())
        with np.load(path) as z:
            d = {k: z[k].copy() for k in ["state", "action", "context", "next_state"]}
        files.append(
            {
                "source": source,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if source == "branch":
            h1.append(d)
            continue
        assert np.array_equal(d["state"][1:], d["next_state"][:-1])
        for start in range(len(d["state"]) - 7):
            sequences.append(
                {
                    k: (d[k][start] if k == "state" else d[k][start : start + 8])
                    for k in d
                }
            )
assert len(sequences) == 3249 and len(h1) == 20
recorded = {
    (x["source"], x["file"]): x["sha256"]
    for x in meta["sources"]
    if x["split"] == "fit"
}
assert recorded == {(x["source"], x["file"]): x["sha256"] for x in files}


def tensor(x):
    return torch.as_tensor(x, device="cuda:0", dtype=torch.float32)


with torch.no_grad():
    h8_sum = 0.0
    for start in range(0, len(sequences), 128):
        rows = sequences[start : start + 128]
        d = {k: tensor(np.stack([x[k] for x in rows])) for k in rows[0]}
        state = d["state"]
        loss = 0.0
        for j in range(8):
            state = model(state, d["action"][:, j], d["context"][:, j])
            loss += model.model.state_loss(state, d["next_state"][:, j]) / 8
        h8_sum += float(loss) * len(rows)
    h8 = h8_sum / len(sequences)
    d = {k: tensor(np.concatenate([x[k] for x in h1])) for k in h1[0]}
    assert len(d["state"]) == 1200
    h1_loss = float(
        base.state_loss(model(d["state"], d["action"], d["context"]), d["next_state"])
    )
    response = AntitheticResponseLoss(a.root / "orthogonal_branches_v1", a.pair_audit)
    d = response.data
    pred = base(
        d["state"].reshape(-1, 73),
        d["action"].reshape(-1, 30),
        d["context"].reshape(-1, 34),
    ).reshape(240, 4, 73)
    delta = response.derivative(pred) - response.target
    std = (base.state_norm.var + base.state_norm.eps).sqrt()
    std = torch.where(base.state_norm.frozen_mask, torch.ones_like(std), std)
    error = delta / std
    response_loss = float(
        0.5 * (error[:, 10:13].square().mean() + error[:, 43:73].square().mean())
    )
base_loss = 0.5 * (h8 + h1_loss)
assert (
    np.isfinite([base_loss, response_loss]).all()
    and base_loss > 0
    and response_loss > 0
)
coefficient = base_loss / response_loss
for k, v in base.state_dict().items():
    assert torch.equal(v.cpu(), w[k]), k
report = {
    "scope": "Frozen structured parent, all fit windows and branch conditions, no optimization or validation selection.",
    "parent_sha256": plan["parent_sha256"],
    "h8_windows": len(sequences),
    "h1_transitions": 1200,
    "response_conditions": 240,
    "h8_loss": h8,
    "h1_loss": h1_loss,
    "base_loss": base_loss,
    "response_loss": response_loss,
    "coefficient": coefficient,
    "sources": files,
    "normalizers_and_weights_unchanged": True,
}
(a.output / "measurement.json").write_text(
    json.dumps(report, indent=2, allow_nan=False)
)
print(json.dumps({k: v for k, v in report.items() if k != "sources"}))
