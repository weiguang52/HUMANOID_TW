#!/usr/bin/env python3
"""Build a full motion manifest weighted by fixed per-motion evaluation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_weighted_manifest(
    source: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    source_path: Path,
    evaluation_path: Path,
    full_weight: float,
    partial_weight: float,
    zero_weight: float,
) -> dict[str, Any]:
    weights = (full_weight, partial_weight, zero_weight)
    if not all(math.isfinite(value) and value > 0.0 for value in weights):
        raise ValueError(f"All weights must be finite and positive, got {weights}")
    if not zero_weight >= partial_weight >= full_weight:
        raise ValueError("Weights must satisfy zero >= partial >= full")

    motions = source.get("motions")
    rows = evaluation.get("motions")
    if source.get("schema_version") != 1 or not isinstance(motions, list) or not motions:
        raise ValueError("Source must be a non-empty schema-v1 motion manifest")
    if evaluation.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError("Evaluation must be a schema-v1 result")
    expected_indices = list(range(len(motions)))
    if evaluation.get("selected_motion_indices") != expected_indices:
        raise ValueError("Evaluation must cover every source motion exactly once in manifest order")
    if len(rows) != len(motions):
        raise ValueError(f"Evaluation row count {len(rows)} != source motion count {len(motions)}")
    expected_sha = evaluation.get("manifest_sha256")
    actual_sha = sha256(source_path)
    if expected_sha != actual_sha:
        raise ValueError(f"Evaluation manifest SHA mismatch: expected={expected_sha}, actual={actual_sha}")

    episodes_per_motion = int(evaluation.get("episodes_per_motion", 0))
    if episodes_per_motion <= 0:
        raise ValueError("Evaluation episodes_per_motion must be positive")
    copied_motions: list[dict[str, Any]] = []
    classes: Counter[str] = Counter()
    for index, (motion, row) in enumerate(zip(motions, rows)):
        motion_id = str(motion.get("id", ""))
        if row.get("motion_index") != index or row.get("motion_id") != motion_id:
            raise ValueError(f"Evaluation row {index} does not match source motion {motion_id}")
        if int(row.get("episodes", -1)) != episodes_per_motion:
            raise ValueError(f"Evaluation row {motion_id} has an unexpected episode count")
        successes = int(row.get("successes", -1))
        if successes < 0 or successes > episodes_per_motion:
            raise ValueError(f"Invalid success count for {motion_id}: {successes}")
        if successes == 0:
            label, weight = "zero_success", zero_weight
        elif successes < episodes_per_motion:
            label, weight = "partial_success", partial_weight
        else:
            label, weight = "full_success", full_weight
        copied = dict(motion)
        copied["weight"] = weight
        copied["evaluation_successes"] = successes
        copied["evaluation_episodes"] = episodes_per_motion
        copied["evaluation_success_rate"] = successes / episodes_per_motion
        copied["evaluation_weight_class"] = label
        copied_motions.append(copied)
        classes[label] += 1

    result = {key: value for key, value in source.items() if key not in {"motions", "curriculum"}}
    result["motions"] = copied_motions
    result["curriculum"] = {
        "name": "practice9_evaluation_weighted_v1",
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": actual_sha,
        "evaluation_results": str(evaluation_path.resolve()),
        "evaluation_results_sha256": sha256(evaluation_path),
        "checkpoint": evaluation.get("checkpoint"),
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "episodes_per_motion": episodes_per_motion,
        "class_counts": dict(sorted(classes.items())),
        "weights": {
            "full_success": full_weight,
            "partial_success": partial_weight,
            "zero_success": zero_weight,
        },
        "motion_count": len(copied_motions),
        "frames": sum(int(motion["frames"]) for motion in copied_motions),
    }
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-weight", type=float, default=1.0)
    parser.add_argument("--partial-weight", type=float, default=2.0)
    parser.add_argument("--zero-weight", type=float, default=4.0)
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    evaluation_path = args.evaluation_results.expanduser().resolve()
    result = build_weighted_manifest(
        read_json(source_path),
        read_json(evaluation_path),
        source_path=source_path,
        evaluation_path=evaluation_path,
        full_weight=args.full_weight,
        partial_weight=args.partial_weight,
        zero_weight=args.zero_weight,
    )
    atomic_json(args.output, result)
    print(json.dumps(result["curriculum"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
