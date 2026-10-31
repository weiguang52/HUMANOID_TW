"""Independent integrity checks for native root-pose state branches."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
a = p.parse_args()
assert not (a.root / "integrity.json").exists()
plan = json.loads((a.root / "plan.json").read_text())
report = json.loads((a.root / "results.json").read_text())
assert report["contract"] == plan["contract"] == "root_pose_state_branches_v1"
assert set(plan["fit_warmups"]).isdisjoint(plan["validation_warmups"])
expected = (len(plan["fit_warmups"]) + len(plan["validation_warmups"])) * (
    1 + len(plan["levels"]) * 2
)
assert len(report["rows"]) == expected
files = []
max_write = 0.0
max_replay = 0.0
terminated = 0
loaded = {}
for r in report["rows"]:
    path = (a.root / r["file"]).resolve()
    assert path.is_relative_to(a.root.resolve())
    with np.load(path) as z:
        d = {k: z[k].copy() for k in z.files}
    for k, v in d.items():
        assert np.isfinite(v).all(), k
    for k, width in [
        ("state", 73),
        ("context", 34),
        ("action", 30),
        ("next_state", 73),
    ]:
        assert d[k].shape == (12, width)
    assert np.count_nonzero(d["action"]) == 0
    assert r["warmup"] in plan[r["split"] + "_warmups"]
    key = (r["split"], r["warmup"], r["kind"], r.get("level"), r.get("assignment"))
    assert key not in loaded
    loaded[key] = d
    terminated += int(d["terminated"].sum())
    if r["kind"] == "branch":
        axis = np.repeat(np.arange(6), 2)
        sign = np.tile([-1, 1], 6) * (1 if r["assignment"] == 0 else -1)
        np.testing.assert_array_equal(d["axis"], axis)
        np.testing.assert_array_equal(d["sign"], sign)
        requested = np.zeros((12, 6))
        requested[np.arange(12), axis] = (
            np.array(plan["base_delta"])[axis] * sign * r["level"]
        )
        np.testing.assert_allclose(d["requested_delta"], requested, rtol=0, atol=1e-8)
        expected_state = d["zero_state"].copy()
        expected_state[:, :3] += d["requested_delta"][:, :3]
        rv = d["requested_delta"][:, 3:].astype(float)
        angle = np.linalg.norm(rv, axis=1, keepdims=True)
        dq = np.concatenate(
            [np.cos(angle / 2), 0.5 * np.sinc(angle / (2 * np.pi)) * rv], axis=1
        )
        q = d["zero_state"][:, 3:7].astype(float)
        product = np.concatenate(
            [
                dq[:, :1] * q[:, :1]
                - np.sum(dq[:, 1:] * q[:, 1:], axis=1, keepdims=True),
                dq[:, :1] * q[:, 1:]
                + q[:, :1] * dq[:, 1:]
                + np.cross(dq[:, 1:], q[:, 1:]),
            ],
            axis=1,
        )
        expected_state[:, 3:7] = product / np.linalg.norm(
            product, axis=1, keepdims=True
        )
        error = float(np.max(np.abs(d["state"] - expected_state)))
        assert error <= 2e-6
        max_write = max(max_write, error)
        max_replay = max(max_replay, r["max_replay_state_error"])
        assert r["max_replay_state_error"] <= plan["replay_atol"]
    files.append(
        {"file": r["file"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
for key, d in loaded.items():
    if key[2] == "branch":
        z = loaded[(key[0], key[1], "zero", None, None)]
        np.testing.assert_allclose(d["context"], z["context"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            d["zero_state"], z["state"], rtol=0, atol=plan["replay_atol"]
        )
        np.testing.assert_array_equal(d["zero_next_state"], z["next_state"])
result = {
    "passed": True,
    "contract": plan["contract"],
    "files": files,
    "max_requested_state_error": max_write,
    "max_replay_error": max_replay,
    "terminated_count": terminated,
    "scope": "Archive validity and requested world-frame perturbations only; no prediction, stability, or source-generalization acceptance.",
}
(a.root / "integrity.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "files"}))
