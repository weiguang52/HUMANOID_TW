"""Audit actual migration smoke/resume outputs, not only shell status."""

import argparse, json, hashlib
from pathlib import Path
import torch, yaml

p = argparse.ArgumentParser()
p.add_argument("state", type=Path)
p.add_argument("mode", choices=["smoke", "reload"])
a = p.parse_args()
s = a.state
mode = a.mode
out = s / ("actual_" + mode)
assert (out / "status").read_text().strip() == "trained"
cp = Path((out / "final_checkpoint").read_text().strip())
run = cp.parent
x = torch.load(cp, map_location="cpu", weights_only=False)
initial = Path((out / "initial_checkpoint").read_text().strip())
old = torch.load(initial, map_location="cpu", weights_only=False)
assert x["iter"] == 27996
for kind in ["actor_state_dict", "critic_state_dict"]:
    assert all(torch.isfinite(v).all() for v in x[kind].values())
assert x["actor_state_dict"]["mlp.0.weight"].shape[1] == 203
steps = sorted(
    set(
        float(v["step"])
        for v in x["optimizer_state_dict"]["state"].values()
        if "step" in v
    )
)
expected = 560040 if mode == "smoke" else 560060
assert steps == [expected], steps
changes = {}
expansions = []
for k, v in old["actor_state_dict"].items():
    new = x["actor_state_dict"][k]
    if new.shape != v.shape:
        assert (
            k == "mlp.0.weight"
            and v.shape[1] == 195
            and new.shape[1] == 203
            and new.shape[0] == v.shape[0]
        )
        expansions.append({"name": k, "old_inputs": 195, "new_inputs": 203})
        new = new[:, :195]
    changes[k] = float((new - v).abs().max())
assert max(changes.values()) > 0
w = json.load(open(run / "physics_slip_completed.json"))
assert w == {
    "completed_windows": 48 if mode == "smoke" else 24,
    "unconsumed_samples": 0,
}
env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
assert env["scene"]["num_envs"] == "64"
assert agent["max_iterations"] == ("2" if mode == "smoke" else "1")
binding = json.load(open(run / "params/training_speed_blend.json"))
assert (
    binding["version"] == 3
    and binding["candidate_action_scope"]["scope"] == "lower_body_waist"
)
if mode == "reload":
    assert initial == Path((s / "actual_smoke/final_checkpoint").read_text().strip())
result = {
    "passed": True,
    "mode": mode,
    "checkpoint": str(cp),
    "sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
    "initial_checkpoint": str(initial),
    "adam_steps": steps,
    "physics": w,
    "max_actor_weight_change": max(changes.values()),
    "expected_input_expansions": expansions,
    "num_envs": 64,
    "actor_obs_dim": 203,
    "scope": "actual training/update/resume proof, not convergence or policy quality",
}
(s / (mode + "_migration_audit.json")).write_text(json.dumps(result, indent=2))
print(result)
