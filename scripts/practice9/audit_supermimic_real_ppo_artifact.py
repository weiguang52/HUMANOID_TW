"""Audit the fixed real-feedback PPO budget and serialized training evidence.

This is an artifact-integrity check, never a physical-control acceptance gate.
It reads CPU tensors only and does not initialize or modify a simulator/policy.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

KIND = "real_physics_ppo_diagnostic_policy_v1"
PARAMETER_PREFIXES = ("encoder.", "decoder.", "critic.")
NORMALIZER_PREFIXES = ("obs_norm.", "goal_norm.", "critic_norm.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if torch.is_tensor(value):
        assert bool(torch.isfinite(value).all()), "Non-finite serialized tensor"
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value), "Non-finite scalar"


def audit(run, parent, manifest_path, expected_updates, expected_envs):
    path = run / "checkpoint_final.pt"
    source = torch.load(parent, map_location="cpu", weights_only=False)
    final = torch.load(path, map_location="cpu", weights_only=False)
    assert final["kind"] == KIND
    plan = final["plan"]
    assert plan == json.loads((run / "plan.json").read_text())
    assert plan["parent_sha256"] == digest(parent)
    assert plan["updates"] == expected_updates and plan["num_envs"] == expected_envs
    assert (
        plan["rollout_steps"] == 32 and plan["epochs"] == 4 and plan["minibatches"] == 4
    )
    assert plan["seed"] == 42 and plan["learning_rate"] == 1e-4
    assert plan["noise_std"] == source["cfg"]["action_noise"] == 0.1
    assert (
        plan["gamma"] == source["cfg"]["gamma"] == 0.99
        and plan["lam"] == 0.95
        and plan["clip"] == 0.2
    )
    assert plan["fixed_source_normalizers"] and plan["all_actions_trainable"]
    assert plan["world_model_used_for_policy_update"] is False
    assert final["cfg"]["extra"] == source["cfg"]["extra"]
    for key, value in source["cfg"].items():
        if key.startswith(("reward_", "sensor_noise_")):
            assert final["cfg"][key] == value, key
    for filename, expected in plan["motion_sha256"].items():
        assert digest(filename) == expected, filename
    assert digest(manifest_path) == plan["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert {item["file"] for item in manifest["motions"]} == set(plan["motion_sha256"])
    expected_transitions = expected_envs * 32 * expected_updates
    assert plan["actual_transitions"] == expected_transitions
    assert final["ppo_update"] == expected_updates
    assert final["iteration"] == source["iteration"] + expected_updates
    assert final["env_steps"] - source["env_steps"] == expected_transitions
    assert not any(
        key in final for key in ("wm_optimizer", "actor_optimizer", "critic_optimizer")
    )
    finite(final)
    assert final["world_model"].keys() == source["world_model"].keys()
    for key, tensor in source["world_model"].items():
        assert torch.equal(final["world_model"][key], tensor), key
    actor = final["actor_critic"]
    assert actor.keys() == source["actor_critic"].keys()
    normalizers = [key for key in actor if key.startswith(NORMALIZER_PREFIXES)]
    assert normalizers
    for key in normalizers:
        assert torch.equal(actor[key], source["actor_critic"][key]), key
    parameters = {
        key: tensor
        for key, tensor in actor.items()
        if key.startswith(PARAMETER_PREFIXES)
    }
    assert parameters and all(key.endswith((".weight", ".bias")) for key in parameters)
    optimizer = final["ppo_optimizer"]
    groups = optimizer["param_groups"]
    assert len(groups) == 1 and groups[0]["lr"] == 1e-4
    ids = groups[0]["params"]
    assert (
        len(ids) == len(set(ids)) == len(parameters) + 1
    )  # extra fixed std has no Adam state
    states = optimizer["state"]
    assert len(states) == len(parameters) and set(states).issubset(ids)
    expected_optimizer_steps = expected_updates * 4 * 4
    for state in states.values():
        assert float(state["step"]) == expected_optimizer_steps
        assert state["exp_avg"].shape == state["exp_avg_sq"].shape
    assert Counter(tuple(value.shape) for value in parameters.values()) == Counter(
        tuple(state["exp_avg"].shape) for state in states.values()
    )
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    finite(rows)
    # Duplicate/reordered updates after a failed attempt require explicit review,
    # rather than silently selecting a successful-looking log segment.
    assert [row["update"] for row in rows] == list(range(1, expected_updates + 1))
    for row in rows:
        assert (
            row["env_steps"] - source["env_steps"] == row["update"] * expected_envs * 32
        )
        if "diagnostic_env_steps" in row:
            assert row["diagnostic_env_steps"] == row["update"] * expected_envs * 32
        assert row["reset_live_max_error"] <= 1e-6
        assert set(row["losses"]) == {"value", "surrogate", "entropy"}
    training_audit = json.loads((run / "audit.json").read_text())
    assert training_audit["checkpoint_sha256"] == digest(path)
    assert training_audit["completed_updates"] == expected_updates
    assert training_audit["world_model_calls"] == 0
    for key in (
        "adapter_parent_equivalence",
        "world_model_unchanged",
        "normalizers_frozen",
        "actual_ppo_optimizer_roundtrip",
        "timeout_bootstrap_unit_check",
    ):
        assert training_audit[key] is True
    group_changes = {}
    for prefix in PARAMETER_PREFIXES:
        values = [
            (key, tensor)
            for key, tensor in parameters.items()
            if key.startswith(prefix)
        ]
        group_changes[prefix] = {
            "changed_tensors": sum(
                not torch.equal(value, source["actor_critic"][key])
                for key, value in values
            ),
            "total_tensors": len(values),
        }
    return {
        "passed": True,
        "scope": "Serialized training integrity only; physical standing, seven-motion and multi-seed acceptance are unproven",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(path),
        "parent_sha256": digest(parent),
        "updates": expected_updates,
        "real_transitions": expected_transitions,
        "adam_steps_per_trainable_tensor": expected_optimizer_steps,
        "trainable_parameter_tensors": len(parameters),
        "normalizer_tensors": len(normalizers),
        "world_model_unchanged": True,
        "source_parameter_changes": group_changes,
        "manifest_hash_checked": manifest_path.exists(),
        "completed_training_segments_have_unique_contiguous_updates": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument("--expected-envs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve prior audits"
    result = audit(
        args.run, args.parent, args.manifest, args.expected_updates, args.expected_envs
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
