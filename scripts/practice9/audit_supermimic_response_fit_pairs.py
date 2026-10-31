"""Verify fit-only antithetic action-response pairs before any optimization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--branches", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
plan = json.loads((a.branches / "plan.json").read_text())
assert set(plan["fit_warmups"]).isdisjoint(plan["validation_warmups"])
integrity = json.loads((a.branches / "integrity.json").read_text())
assert integrity["passed"]
hashes = {r["file"]: r["sha256"] for r in integrity["files"]}
rows = json.loads((a.branches / "results.json").read_text())["rows"]
loaded = {}
sources = []
for row in rows:
    if row["split"] != "fit":
        continue
    path = (a.branches / row["file"]).resolve()
    assert path.is_relative_to(a.branches.resolve())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == hashes[row["file"]]
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    for key, width in [
        ("state", 73),
        ("action", 30),
        ("context", 34),
        ("next_state", 73),
    ]:
        assert d[key].shape == (60, width) and np.isfinite(d[key]).all()
    loaded[(row["warmup"], row["kind"], row.get("epsilon"), row.get("assignment"))] = d
    sources.append({"file": row["file"], "sha256": sha})
assert len(loaded) == 20
conditions = []
targets = []
max_state_error = 0.0
for warmup in plan["fit_warmups"]:
    zero = loaded[(warmup, "zero", None, None)]
    for eps in plan["epsilons"]:
        pair = [loaded[(warmup, "branch", eps, i)] for i in [0, 1]]
        for assignment, d in enumerate(pair):
            np.testing.assert_array_equal(d["joint"], np.repeat(np.arange(30), 2))
            signs = np.tile([-1, 1] if assignment == 0 else [1, -1], 30)
            np.testing.assert_array_equal(d["sign"], signs)
            expected = np.zeros((60, 30))
            expected[np.arange(60), d["joint"]] = signs * eps
            np.testing.assert_allclose(d["action"], expected, atol=1e-8, rtol=0)
            assert not d["terminated"].any()
            err = float(np.max(np.abs(d["state"] - zero["state"])))
            max_state_error = max(max_state_error, err)
            assert err <= plan["replay_atol"]
            np.testing.assert_allclose(d["context"], zero["context"], atol=1e-7, rtol=0)
        n0, n1 = [d["next_state"].astype(float) for d in pair]
        derivative = (n0[1::2] - n0[::2] + n1[::2] - n1[1::2]) / (4 * eps)
        target = np.concatenate([derivative[:, 10:13], derivative[:, 43:73]], axis=1)
        assert target.shape == (30, 33) and np.isfinite(target).all()
        targets.append(target)
        conditions.append(
            {
                "warmup": warmup,
                "epsilon": eps,
                "axes": 30,
                "root_omega_response_norm": float(np.linalg.norm(target[:, :3])),
                "joint_velocity_response_norm": float(np.linalg.norm(target[:, 3:])),
            }
        )
targets = np.concatenate(targets)
report = {
    "passed": True,
    "scope": "Fit-only 240 axis conditions, four endpoints each; not temporal episodes, independent source validation, or model success. No optimizer ran.",
    "fit_conditions": len(targets),
    "max_state_replay_error": max_state_error,
    "source_files": sources,
    "conditions": conditions,
    "target_rms_by_component": np.sqrt(np.mean(targets**2, axis=0)).tolist(),
    "training_started": False,
}
a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(
    json.dumps(
        {
            "passed": True,
            "fit_conditions": len(targets),
            "max_state_replay_error": max_state_error,
        }
    )
)
