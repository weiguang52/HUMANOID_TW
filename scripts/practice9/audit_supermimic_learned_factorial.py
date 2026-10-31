"""Read-only verification of frozen-policy factorial first-episode archives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    a = p.parse_args()
    r = a.root
    out = r / "independent_integrity.json"
    assert not out.exists(), "Preserve prior audits"
    d = json.loads((r / "results.json").read_text())
    plan = d["plan"]
    assert plan == json.loads((r / "plan.json").read_text())
    assert digest(a.checkpoint) == plan["checkpoint_sha256"]
    source = json.loads((r / "source_config.json").read_text())
    extra = source["extra"]
    cap = np.array(
        [extra["tw_joint_speed_limits"][n] for n in extra["tw_state_joint_names"]]
    )
    profiles = [x[0] for x in plan["profiles"]]
    agents = plan["agents"]
    b = plan["num_envs"]
    steps = plan["max_steps"]
    expected = {
        (name, agent, i) for name in profiles for agent in agents for i in range(b)
    }
    rows = {(x["profile"], x["agent"], x["env_id"]): x for x in d["rows"]}
    assert len(rows) == len(d["rows"]) and set(rows) == expected
    assert len(d["summary"]) == len(profiles) * len(agents)
    with np.load(r / "fixed_noise.npz", allow_pickle=False) as noise:
        action = noise["action_noise"].copy()
        sensor = noise["sensor_noise"]
        scale = noise["sensor_scale"]
        assert action.shape == (steps, b, 30) and sensor.shape == (steps, b, 96)
        assert np.isfinite(action).all() and np.isfinite(sensor).all()
        assert (np.abs(sensor) <= scale[None, None] + 1e-7).all()
        assert not sensor[:, :, 66:].any()
    initials = {}
    max_initial_error = 0.0
    max_speed_excess = 0.0
    for profile in profiles:
        reset = profile.startswith("r1")
        for agent in agents:
            with np.load(r / f"{profile}_{agent}_initial.npz", allow_pickle=False) as z:
                initial = {k: z[k].copy() for k in z.files}
            if reset in initials:
                err = float(np.max(np.abs(initial["state"] - initials[reset]["state"])))
                max_initial_error = max(max_initial_error, err)
                assert err <= 1e-6
                np.testing.assert_allclose(
                    initial["context"], initials[reset]["context"], rtol=0, atol=1e-7
                )
            else:
                initials[reset] = initial
            for i in range(b):
                row = rows[profile, agent, i]
                n = row["steps"]
                assert 0 < n <= steps and abs(row["seconds"] - n * 0.02) < 1e-10
                assert row["completed_requested_window"] == (
                    n == steps and not row["termination_reasons"]
                )
                assert row["completed20s"] == (
                    n == 1000 and not row["termination_reasons"]
                )
                if n < steps:
                    assert row["termination_reasons"]
                with np.load(r / row["trace"], allow_pickle=False) as z:
                    for key, width in [
                        ("state", 73),
                        ("next_state", 73),
                        ("action", 30),
                        ("context", 34),
                        ("foot_tilt_deg", 2),
                    ]:
                        assert (
                            z[key].shape == (n, width) and np.isfinite(z[key]).all()
                        ), key
                    assert np.array_equal(z["next_state"][:-1], z["state"][1:])
                    np.testing.assert_allclose(
                        z["state"][0], initial["state"][i], rtol=0, atol=0
                    )
                    excess = float(
                        np.max(np.abs(z["next_state"][:, 43:73]) - cap[None])
                    )
                    max_speed_excess = max(max_speed_excess, excess)
                    assert excess <= 1e-5
                    if agent == "zero":
                        expected_action = (
                            action[:n, i]
                            if profile.endswith("a1")
                            else np.zeros_like(z["action"])
                        )
                        assert np.array_equal(z["action"], expected_action)
    sensor_control_error = 0.0
    for profile in profiles:
        if "_s0_" not in profile:
            continue
        other = profile.replace("_s0_", "_s1_")
        if other not in profiles:
            continue
        for i in range(b):
            left = rows[profile, "zero", i]
            right = rows[other, "zero", i]
            assert (
                left["steps"] == right["steps"]
                and left["termination_reasons"] == right["termination_reasons"]
            )
            with np.load(r / left["trace"]) as x, np.load(r / right["trace"]) as y:
                err = float(np.max(np.abs(x["state"] - y["state"])))
                sensor_control_error = max(sensor_control_error, err)
                # Preserve the strict numeric check as a reported failure.
                # GPU contact repeats can differ; structural archive checks continue.
    for summary in d["summary"]:
        group = [
            x
            for x in d["rows"]
            if x["profile"] == summary["profile"] and x["agent"] == summary["agent"]
        ]
        assert len(group) == b and summary["episodes"] == b
        assert summary["completed20s"] == sum(x["completed20s"] for x in group)
        assert summary["parking_max_active_state_error"] <= 1e-6
        assert summary["weights_frozen"] and summary["world_model_calls"] == 0
    result = {
        "passed": sensor_control_error <= 1e-5,
        "archive_invariants_passed": True,
        "numeric_pairing_passed": sensor_control_error <= 1e-5,
        "numeric_pairing_atol": 1e-5,
        "episodes": len(rows),
        "profile_agent_groups": len(d["summary"]),
        "max_initial_state_mismatch": max_initial_error,
        "max_joint_speed_excess": max_speed_excess,
        "pd_sensor_toggle_max_state_difference": sensor_control_error,
        "checkpoint_sha256": digest(a.checkpoint),
        "results_sha256": digest(r / "results.json"),
        "fixed_noise_sha256": digest(r / "fixed_noise.npz"),
        "scope": "Archive integrity and matched controls only; one policy training seed, not full independent-motion/multi-seed acceptance",
    }
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(
            "Archive invariants passed; strict numeric pairing failed; report preserved"
        )


if __name__ == "__main__":
    main()
