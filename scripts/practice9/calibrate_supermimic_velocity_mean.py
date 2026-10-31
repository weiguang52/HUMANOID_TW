"""Analytic fit-only velocity mean correction; no new gradient updates."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from supermimic.algo.networks import WorldModel, WorldModelCfg
from supermimic_pose_integration_diagnostic import POSE_KIND

p = argparse.ArgumentParser(description=__doc__)
for k in ["parent", "branches", "output", "plan"]:
    p.add_argument("--" + k, type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
a.output.mkdir(parents=True)
preregistered = json.loads(a.plan.read_text())
assert (
    preregistered["source_sha256"] == hashlib.sha256(a.parent.read_bytes()).hexdigest()
)
c = torch.load(a.parent, map_location="cpu", weights_only=False)
assert c["kind"] == POSE_KIND and c["calibration_updates"] == 1000
w = c["world_model"]
cfg = c["cfg"]
mapping = json.loads((a.branches / "control_mapping.json").read_text())
assert (
    mapping["state_joint_names"] == cfg["extra"]["tw_state_joint_names"]
    and mapping["action_joint_names"] == cfg["extra"]["tw_action_joint_names"]
)
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
parts = []
files = []
for row in json.loads((a.branches / "results.json").read_text())["rows"]:
    if row["split"] != "fit":
        continue
    path = (a.branches / row["file"]).resolve()
    assert path.is_relative_to(a.branches.resolve())
    with np.load(path) as z:
        parts.append(
            {k: z[k].copy() for k in ["state", "action", "context", "next_state"]}
        )
    files.append(
        {"file": row["file"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
assert len(files) == 20
source_inputs = json.loads((a.parent.parent / "fit_input.json").read_text())
assert {x["file"]: x["sha256"] for x in files} == {
    x["file"]: x["sha256"] for x in source_inputs["one_step_files"]
}
x = {
    k: torch.as_tensor(np.concatenate([d[k] for d in parts]), device="cuda:0")
    for k in parts[0]
}
assert len(x["state"]) == 1200
velocity_indices = list(range(7, 13)) + list(range(43, 73))
output_indices = list(range(3, 9)) + list(range(39, 69))
with torch.no_grad():
    for _ in range(5):
        base(x["state"], x["action"], x["context"])
    before = base(x["state"], x["action"], x["context"])
    error = before[:, velocity_indices] - x["next_state"][:, velocity_indices]
    correction = -error.double().mean(dim=0).float()
    assert torch.isfinite(correction).all()
    base.net[-1].bias[output_indices] += correction
    after = base(x["state"], x["action"], x["context"])
    torch.testing.assert_close(
        after[:, velocity_indices] - before[:, velocity_indices],
        correction.expand(1200, -1),
        atol=1e-6,
        rtol=0,
    )
    untouched = [i for i in range(73) if i not in velocity_indices]
    assert torch.equal(before[:, untouched], after[:, untouched])
    after_error = after[:, velocity_indices] - x["next_state"][:, velocity_indices]
    assert float(after_error.square().mean()) <= float(error.square().mean())
new = {k: v.cpu().clone() for k, v in base.state_dict().items()}
changed = []
for k, v in new.items():
    assert torch.isfinite(v).all()
    if not torch.equal(v, w[k]):
        changed.append(k)
assert changed == ["net.6.bias"], changed
mask = torch.ones_like(new["net.6.bias"], dtype=torch.bool)
mask[output_indices] = False
assert torch.equal(new["net.6.bias"][mask], w["net.6.bias"][mask])
result = copy.deepcopy(c)
result["world_model"] = new
result.pop("optimizer", None)
result["pre_correction_calibration_plan"] = copy.deepcopy(c["calibration_plan"])
source_hash = hashlib.sha256(a.parent.read_bytes()).hexdigest()
operation = {
    "method": "negative fitH1 mean physical velocity error, all1200 fit transitions",
    "new_optimizer_updates": 0,
    "underlying_source_updates": 1000,
    "source_sha256": source_hash,
    "velocity_state_indices": velocity_indices,
    "output_bias_indices": output_indices,
    "correction": correction.cpu().tolist(),
    "fit_files": files,
    "fit_velocity_mse_before": float(error.square().mean()),
    "fit_velocity_mse_after": float(after_error.square().mean()),
    "scope": "Analytical post-fit diagnostic, not policy promotion. All other tensors and explicit prediction contract preserved; no stale optimizer retained.",
}
result["postfit_calibration"] = operation
result["calibration_plan"].update(
    source_checkpoint_sha256=source_hash,
    operation="analytic_velocity_bias_postfit",
    differing_variable="Analytical36 velocity output bias corrections only; no new optimizer updates",
    new_optimizer_updates=0,
)
torch.save(result, a.output / "world_model_calibrated.pt")
(a.output / "fit_input.json").write_bytes(
    (a.parent.parent / "fit_input.json").read_bytes()
)
(a.output / "bias_fit_input.json").write_text(json.dumps(operation, indent=2))
(a.output / "status").write_text("corrected_pending_independent_audit\n")
print(
    json.dumps(
        {k: v for k, v in operation.items() if k not in ["fit_files", "correction"]}
    )
)
