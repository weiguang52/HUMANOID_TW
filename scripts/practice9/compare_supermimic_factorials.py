"""Compare every fixed factorial group, without selecting episodes or policies."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), "Preserve previous comparisons"
    roots = [a.before, a.after]
    reports = [json.loads((r / "results.json").read_text()) for r in roots]
    audits = [json.loads((r / "independent_integrity.json").read_text()) for r in roots]
    assert all(x["archive_invariants_passed"] for x in audits)
    plans = [x["plan"] for x in reports]
    for key in [
        "seed",
        "action_noise_seed",
        "sensor_noise_seed",
        "profiles",
        "agents",
        "num_envs",
        "max_steps",
        "manifest_sha256",
        "motion_sha256",
        "events_disabled",
        "physics_contract",
    ]:
        assert plans[0][key] == plans[1][key], key
    with (
        np.load(roots[0] / "fixed_noise.npz") as x,
        np.load(roots[1] / "fixed_noise.npz") as y,
    ):
        assert x.files == y.files and all(
            np.array_equal(x[key], y[key]) for key in x.files
        )
    configs = [json.loads((r / "source_config.json").read_text()) for r in roots]
    for key in [
        "action_noise",
        "sensor_noise_joint_pos",
        "sensor_noise_joint_vel",
        "sensor_noise_base_ang_vel",
        "sensor_noise_projected_gravity",
    ]:
        assert configs[0][key] == configs[1][key], key
    for key in [
        "tw_mjcf_sha256",
        "tw_physics_contract",
        "tw_joint_speed_limits",
        "tw_state_joint_names",
        "tw_action_joint_names",
    ]:
        assert configs[0]["extra"][key] == configs[1]["extra"][key], key
    maximum = 0.0
    for profile in plans[0]["profiles"]:
        for agent in plans[0]["agents"]:
            with (
                np.load(roots[0] / f"{profile[0]}_{agent}_initial.npz") as x,
                np.load(roots[1] / f"{profile[0]}_{agent}_initial.npz") as y,
            ):
                err = float(np.max(np.abs(x["state"] - y["state"])))
                maximum = max(maximum, err)
                assert err <= 1e-6
                np.testing.assert_allclose(
                    x["context"], y["context"], rtol=0, atol=1e-7
                )
    rowmaps = [
        {(x["profile"], x["agent"], x["env_id"]): x for x in d["rows"]} for d in reports
    ]
    assert rowmaps[0].keys() == rowmaps[1].keys()
    summaries = [{(x["profile"], x["agent"]): x for x in d["summary"]} for d in reports]
    assert summaries[0].keys() == summaries[1].keys()
    rows = []
    for key in summaries[0]:
        before = summaries[0][key]
        after = summaries[1][key]
        rows.append(
            {
                "profile": key[0],
                "agent": key[1],
                "before_completed20s": before["completed20s"],
                "after_completed20s": after["completed20s"],
                "episodes": before["episodes"],
                "before_median_seconds": before["median_seconds"],
                "after_median_seconds": after["median_seconds"],
            }
        )
    pd_ending_differences = [
        key
        for key in rowmaps[0]
        if key[1] == "zero"
        and (
            rowmaps[0][key]["steps"] != rowmaps[1][key]["steps"]
            or rowmaps[0][key]["termination_reasons"]
            != rowmaps[1][key]["termination_reasons"]
        )
    ]
    result = {
        "scope": "One policy training seed; all fixed developer episodes retained, not statistical generalization or final-motion acceptance",
        "inputs_matched": True,
        "noise_arrays_bit_identical": True,
        "max_initial_state_difference": maximum,
        "strict_numeric_pairing_passed": [x["numeric_pairing_passed"] for x in audits],
        "pd_ending_differences": pd_ending_differences,
        "before_checkpoint_sha256": plans[0]["checkpoint_sha256"],
        "after_checkpoint_sha256": plans[1]["checkpoint_sha256"],
        "rows": rows,
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
