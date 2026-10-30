"""Audit two real SuperMimic checkpoints for finite updates and optimizer progress."""

import argparse
import hashlib
import json
from pathlib import Path
import torch


def audit(before_path, after_path):
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    after = torch.load(after_path, map_location="cpu", weights_only=False)
    assert after["iteration"] > before["iteration"]
    assert after["env_steps"] > before["env_steps"]
    result = {
        "passed": True,
        "before": str(before_path),
        "after": str(after_path),
        "before_sha256": hashlib.sha256(before_path.read_bytes()).hexdigest(),
        "after_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest(),
        "iterations": [before["iteration"], after["iteration"]],
        "env_steps": [before["env_steps"], after["env_steps"]],
        "components": {},
    }
    for component, prefix in [
        ("world_model", "net."),
        ("actor_critic", "encoder."),
        ("actor_critic", "decoder."),
        ("actor_critic", "critic."),
    ]:
        old, new = before[component], after[component]
        assert old.keys() == new.keys()
        for tensor in new.values():
            if torch.is_floating_point(tensor):
                assert torch.isfinite(tensor).all(), (component, "non-finite")
        delta = max(
            float((new[key] - old[key]).abs().max())
            for key in new
            if key.startswith(prefix)
        )
        assert delta > 0, (component, prefix, "no parameter update")
        result["components"][component + "/" + prefix] = {"max_parameter_change": delta}
    for name in ("wm_optimizer", "actor_optimizer", "critic_optimizer"):
        old, new = before[name]["state"], after[name]["state"]
        assert old and old.keys() == new.keys()
        steps = []
        for key in old:
            a, b = int(old[key]["step"]), int(new[key]["step"])
            assert b > a, (name, key, a, b)
            steps.append((a, b))
            for tensor in new[key].values():
                if isinstance(tensor, torch.Tensor):
                    assert torch.isfinite(tensor).all()
        result[name] = {
            "parameter_states": len(steps),
            "step_pairs": sorted(set(steps)),
        }
    result["scope"] = (
        "Actual full-method finite updates and optimizer continuation; not convergence or policy-quality acceptance. Simulator/replay rebuilt on resume."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
