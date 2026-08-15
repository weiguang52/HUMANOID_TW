"""Pure helpers for deterministic per-motion policy evaluation."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ERROR_METRICS = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
)
TERMINATION_TERMS = ("time_out", "motion_end", "anchor_pos", "anchor_ori", "ee_body_pos")


def parse_motion_indices(spec: str | None, motion_count: int) -> list[int]:
    """Parse comma-separated indexes and inclusive ranges without reordering them."""
    if motion_count <= 0:
        raise ValueError(f"motion_count must be positive, got {motion_count}")
    if spec is None or not spec.strip():
        return list(range(motion_count))

    result: list[int] = []
    seen: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"Empty motion index token in {spec!r}")
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not all(piece.strip().isdigit() for piece in pieces):
                raise ValueError(f"Invalid motion index range: {token!r}")
            start, end = (int(piece.strip()) for piece in pieces)
            if end < start:
                raise ValueError(f"Motion index range must be ascending: {token!r}")
            values = range(start, end + 1)
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid motion index: {token!r}")
            values = (int(token),)
        for value in values:
            if value < 0 or value >= motion_count:
                raise ValueError(f"Motion index {value} is outside [0, {motion_count})")
            if value in seen:
                raise ValueError(f"Duplicate motion index: {value}")
            seen.add(value)
            result.append(value)
    if not result:
        raise ValueError("No motion indexes selected")
    return result


def build_report(
    *,
    selected_indices: Sequence[int],
    clip_ids: Sequence[str],
    clip_lengths: Sequence[int],
    fps: float,
    episodes: Sequence[int],
    successes: Sequence[int],
    completion_sums: Sequence[float],
    reward_sums: Sequence[float],
    metric_steps: Sequence[int],
    metric_sums: dict[str, Sequence[float]],
    termination_counts: dict[str, Sequence[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a JSON-ready aggregate and one row per selected motion."""
    count = len(selected_indices)
    vectors: list[tuple[str, Sequence[Any]]] = [
        ("clip_ids", clip_ids),
        ("clip_lengths", clip_lengths),
        ("episodes", episodes),
        ("successes", successes),
        ("completion_sums", completion_sums),
        ("reward_sums", reward_sums),
        ("metric_steps", metric_steps),
    ]
    vectors.extend((f"metric:{name}", values) for name, values in metric_sums.items())
    vectors.extend((f"termination:{name}", values) for name, values in termination_counts.items())
    mismatched = [name for name, values in vectors if len(values) != count]
    if mismatched:
        raise ValueError(f"Evaluation vectors do not match selected motion count {count}: {mismatched}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    rows: list[dict[str, Any]] = []
    for position, motion_index in enumerate(selected_indices):
        episode_count = int(episodes[position])
        success_count = int(successes[position])
        step_count = int(metric_steps[position])
        if episode_count <= 0 or step_count <= 0:
            raise ValueError(f"Motion {motion_index} has no completed episode or metric samples")
        row: dict[str, Any] = {
            "motion_index": int(motion_index),
            "motion_id": str(clip_ids[position]),
            "length_frames": int(clip_lengths[position]),
            "duration_s": float(clip_lengths[position]) / float(fps),
            "episodes": episode_count,
            "successes": success_count,
            "success_rate": success_count / episode_count,
            "mean_completion_fraction": float(completion_sums[position]) / episode_count,
            "mean_reward_per_step": float(reward_sums[position]) / step_count,
            "metric_steps": step_count,
        }
        for name in ERROR_METRICS:
            row[f"mean_{name}"] = float(metric_sums[name][position]) / step_count
        for name in TERMINATION_TERMS:
            row[f"termination_{name}"] = int(termination_counts[name][position])
        rows.append(row)

    total_episodes = sum(int(value) for value in episodes)
    total_successes = sum(int(value) for value in successes)
    total_metric_steps = sum(int(value) for value in metric_steps)
    summary: dict[str, Any] = {
        "motions": count,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "mean_completion_fraction": sum(float(value) for value in completion_sums) / total_episodes,
        "mean_reward_per_step": sum(float(value) for value in reward_sums) / total_metric_steps,
        "metric_steps": total_metric_steps,
        "motions_full_success": sum(row["success_rate"] == 1.0 for row in rows),
        "motions_zero_success": sum(row["success_rate"] == 0.0 for row in rows),
    }
    for name in ERROR_METRICS:
        summary[f"mean_{name}"] = sum(float(value) for value in metric_sums[name]) / total_metric_steps
    for name in TERMINATION_TERMS:
        summary[f"termination_{name}"] = sum(int(value) for value in termination_counts[name])
    return summary, rows


def atomic_write_reports(output_dir: Path, payload: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    """Atomically publish JSON, CSV, and a compact human-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_dir.is_symlink():
        raise ValueError(f"Evaluation output directory must not be a symlink: {output_dir}")

    json_path = output_dir / "results.json"
    csv_path = output_dir / "per_motion.csv"
    summary_path = output_dir / "summary.txt"
    for path in (json_path, csv_path, summary_path):
        if path.is_symlink():
            raise ValueError(f"Refusing to replace symlink: {path}")

    json_tmp = json_path.with_suffix(".json.tmp")
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    summary_tmp = summary_path.with_suffix(".txt.tmp")
    with json_tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fieldnames = list(rows[0]) if rows else []
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    summary = payload["summary"]
    worst = sorted(rows, key=lambda row: (row["success_rate"], row["mean_completion_fraction"]))[:20]
    with summary_tmp.open("w", encoding="utf-8") as stream:
        stream.write(
            f"motions={summary['motions']} episodes={summary['episodes']} "
            f"success_rate={summary['success_rate']:.6f} "
            f"mean_completion={summary['mean_completion_fraction']:.6f}\n"
        )
        stream.write("worst_motions:\n")
        for row in worst:
            stream.write(
                f"  {row['motion_index']:03d} {row['motion_id']} "
                f"success={row['success_rate']:.3f} completion={row['mean_completion_fraction']:.3f}\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(json_tmp, json_path)
    os.replace(csv_tmp, csv_path)
    os.replace(summary_tmp, summary_path)
