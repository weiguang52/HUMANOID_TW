#!/usr/bin/env python3
"""Select deterministic HumanML3D locomotion candidates without copying arrays."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

POSITIVE = re.compile(r"\b(walk(?:s|ed|ing)?|pace(?:s|d|ing)?|march(?:es|ed|ing)?|step(?:s|ped|ping)?)\b", re.IGNORECASE)
NEGATIVE = re.compile(
    r"\b(run(?:s|ning)?|jog(?:s|ging)?|jump(?:s|ed|ing)?|hop(?:s|ped|ping)?|dance(?:s|d|ing)?|"
    r"sit(?:s|ting)?|lie|lies|lying|lay|kneel(?:s|ed|ing)?|crawl(?:s|ed|ing)?|"
    r"kick(?:s|ed|ing)?|punch(?:es|ed|ing)?|throw(?:s|ing)?|catch(?:es|ing)?|"
    r"stairs?|staircase|ladder|climb(?:s|ed|ing)?|chair|table|box|ball|sword|"
    r"drink(?:s|ing)?|eat(?:s|ing)?|pick(?:s|ed|ing)? up|carry|carries|carrying)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    motion_id: str
    source: str
    text: str
    frames: int
    displacement_m: float
    path_length_m: float
    yaw_change_rad: float
    tag: str
    score: float


def captions(path: Path) -> list[str]:
    return [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(text: str, displacement: float, yaw_change: float) -> str:
    lower = text.lower()
    if "backward" in lower or "backwards" in lower:
        return "backward"
    if "sideways" in lower or "side step" in lower or "to the side" in lower:
        return "sideways"
    if abs(yaw_change) >= 0.65 or any(word in lower for word in ("turn", "circle", "arc", "around")):
        return "turning"
    if displacement < 0.12 or any(word in lower for word in ("in place", "same spot", "treadmill")):
        return "in_place"
    return "forward"


def geometry(joints: np.ndarray) -> tuple[float, float, float]:
    root = np.asarray(joints[:, 0], dtype=np.float64)
    horizontal = root[:, (0, 2)]
    displacement = float(np.linalg.norm(horizontal[-1] - horizontal[0]))
    path_length = float(np.linalg.norm(np.diff(horizontal, axis=0), axis=1).sum())
    lateral = np.asarray(joints[:, 2] - joints[:, 1], dtype=np.float64)
    yaw = np.unwrap(np.arctan2(lateral[:, 2], lateral[:, 0]))
    yaw_change = float(yaw[-1] - yaw[0])
    return displacement, path_length, yaw_change


def inspect(source: Path, text_path: Path, min_frames: int, max_frames: int) -> Candidate | None:
    lines = captions(text_path)
    combined = " ".join(lines)
    if not POSITIVE.search(combined) or NEGATIVE.search(combined):
        return None
    joints = np.load(source, mmap_mode="r", allow_pickle=False)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or not min_frames <= joints.shape[0] <= max_frames:
        return None
    if not np.isfinite(joints).all():
        return None
    displacement, path_length, yaw_change = geometry(joints)
    tag = classify(combined, displacement, yaw_change)
    if path_length < 0.08 and tag != "in_place":
        return None
    score = path_length + 0.35 * displacement + 0.10 * min(abs(yaw_change), math.pi)
    return Candidate(
        motion_id=source.stem,
        source=str(source.resolve()),
        text=lines[0],
        frames=int(joints.shape[0]),
        displacement_m=displacement,
        path_length_m=path_length,
        yaw_change_rad=yaw_change,
        tag=tag,
        score=score,
    )


def select_balanced(candidates: list[Candidate], limit: int) -> list[Candidate]:
    quotas = {
        "forward": math.ceil(limit * 0.45),
        "turning": math.ceil(limit * 0.25),
        "backward": math.ceil(limit * 0.12),
        "sideways": math.ceil(limit * 0.10),
        "in_place": math.ceil(limit * 0.08),
    }
    by_tag: dict[str, list[Candidate]] = {key: [] for key in quotas}
    for item in candidates:
        by_tag[item.tag].append(item)
    chosen: list[Candidate] = []
    for tag, quota in quotas.items():
        chosen.extend(sorted(by_tag[tag], key=lambda item: (-item.score, item.motion_id))[:quota])
    selected_ids = {item.motion_id for item in chosen}
    remainder = sorted(
        (item for item in candidates if item.motion_id not in selected_ids),
        key=lambda item: (-item.score, item.motion_id),
    )
    chosen.extend(remainder[: max(0, limit - len(chosen))])
    return sorted(chosen[:limit], key=lambda item: item.motion_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-joints-dir", type=Path, required=True)
    parser.add_argument("--texts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.limit <= 0 or args.min_frames < 9 or args.max_frames < args.min_frames:
        raise ValueError("invalid selection limits")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    candidates: list[Candidate] = []
    for source in sorted(args.new_joints_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].npy")):
        text_path = args.texts_dir / f"{source.stem}.txt"
        if text_path.is_file():
            item = inspect(source, text_path, args.min_frames, args.max_frames)
            if item is not None:
                candidates.append(item)
    selected = select_balanced(candidates, args.limit)
    if len(selected) < args.limit:
        raise RuntimeError(f"only {len(selected)} eligible motions for requested limit {args.limit}")

    links = args.output_dir / "new_joints"
    links.mkdir(parents=True, exist_ok=True)
    for item in selected:
        (links / f"{item.motion_id}.npy").symlink_to(item.source)
    manifest = {
        "schema_version": 1,
        "selection": "semantic_locomotion_balanced_v1",
        "source_new_joints": str(args.new_joints_dir.resolve()),
        "source_texts": str(args.texts_dir.resolve()),
        "eligible_count": len(candidates),
        "selected_count": len(selected),
        "parameters": {"limit": args.limit, "min_frames": args.min_frames, "max_frames": args.max_frames},
        "tag_counts": {tag: sum(item.tag == tag for item in selected) for tag in sorted({x.tag for x in selected})},
        "motions": [asdict(item) for item in selected],
    }
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "selection.tsv").open("w", encoding="utf-8") as handle:
        handle.write("id\ttag\tframes\tdisplacement_m\tpath_length_m\tyaw_change_rad\ttext\n")
        for item in selected:
            handle.write(
                f"{item.motion_id}\t{item.tag}\t{item.frames}\t{item.displacement_m:.6f}\t"
                f"{item.path_length_m:.6f}\t{item.yaw_change_rad:.6f}\t{item.text}\n"
            )
    print(json.dumps({"eligible": len(candidates), "selected": len(selected), "tag_counts": manifest["tag_counts"]}))


if __name__ == "__main__":
    main()
