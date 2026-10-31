"""Create reproducible static-evaluation metrics, contact polygons and review page.

The generated montage still requires visual inspection; this tool never promotes.
"""

import argparse
import hashlib
import html
import io
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon_area(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    p = np.array(lower[:-1] + upper[:-1])
    return float(
        abs(
            np.dot(p[:, 0], np.roll(p[:, 1], -1))
            - np.dot(p[:, 1], np.roll(p[:, 0], -1))
        )
        * 0.5
    )


def episode(root, row):
    n = row["steps"]
    with np.load(root / row["transition_file"], allow_pickle=False) as z:
        assert len(z["state"]) == n and np.array_equal(
            z["next_state"][:-1], z["state"][1:]
        )
        s = z["next_state"]
        q = s[:, 3:7] / np.linalg.norm(s[:, 3:7], axis=-1, keepdims=True)
        xy = np.linalg.norm(s[:, :2] - z["state"][0, :2], axis=1)
        tilt = np.degrees(
            np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
        )
        metric = {
            "agent": row["agent"],
            "steps": n,
            "seconds": n * 0.02,
            "completed": row["completed_requested_window"],
            "termination_reasons": row["termination_reasons"],
            "root_xy_max_mm": float(xy.max() * 1000),
            "root_xy_end_mm": float(xy[-1] * 1000),
            "root_tilt_max_deg": float(tilt.max()),
            "root_tilt_end_deg": float(tilt[-1]),
            "action_abs_max": float(abs(z["action"]).max()),
            "mean_body_error_mm": row["mean_body_error_m"] * 1000,
            "mean_joint_error_deg": float(np.degrees(row["mean_joint_error_rad"])),
        }
    contacts = json.loads((root / row["native_contact_file"]).read_text())["samples"]
    assert len(contacts) == n
    summaries = {}
    for side in ["left", "right"]:
        feet = [x["feet"][side] for x in contacts]
        areas = []
        for f in feet:
            ground = [
                x for x in f["contacts"] if x["ground"] and x["normal_force_n"] > 1e-4
            ]
            pts = np.array([x["pos_w"] for x in ground]).reshape(-1, 3)
            local = (pts - np.array(f["position_w"])) @ np.array(f["rotation_w"])
            areas.append(
                polygon_area(local[:, :2]) / (f["sole_size_m"][0] * f["sole_size_m"][1])
            )
        summaries[side] = {}
        for label, start in [("full", 0), ("last_up_to10s", max(0, n - 500))]:
            f = feet[start:]
            counts = Counter(x["loaded_ground_points"] for x in f)
            summaries[side][label] = {
                "seconds": len(f) * 0.02,
                "loaded_points_hist": dict(counts),
                "tilt_median_deg": float(np.median([x["tilt_deg"] for x in f])),
                "tilt_max_deg": max(x["tilt_deg"] for x in f),
                "contact_polygon_area_ratio_median": float(np.median(areas[start:])),
                "mean_ground_normal_force_n": float(
                    np.mean([x["ground_normal_force_n"] for x in f])
                ),
            }
    metric["contacts"] = summaries
    return metric


def montage(root, row):
    out = Image.new("RGB", (768, 382), "white")
    draw = ImageDraw.Draw(out)
    indices = [0, row["steps"] // 2, row["steps"] - 1]
    for j, view in enumerate(["front", "side"]):
        for i, index in enumerate(indices):
            frame = subprocess.check_output(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    str(index / 50),
                    "-i",
                    str(root / row[view + "_video"]),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=256:171",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-",
                ]
            )
            out.paste(Image.open(io.BytesIO(frame)), (256 * i, 191 * j + 20))
            draw.text(
                (256 * i + 4, 191 * j + 4),
                f"{row['agent']} {view} {(index + 1) / 50:.2f}s",
                fill="black",
            )
    path = root / f"{row['agent']}_review_montage.jpg"
    out.save(path, quality=55)
    return path.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--before", type=Path)
    a = p.parse_args()
    root = a.evaluation.resolve()
    target = root / "standing_review_metrics.json"
    assert not target.exists(), "Preserve prior review artifacts"
    report = json.loads((root / "results.json").read_text())
    media = json.loads((root / "episode_media_integrity.json").read_text())
    assert media["passed"]
    assert (
        media["checkpoint_sha256"]
        == report["checkpoint_sha256"]
        == digest(report["checkpoint"])
    )
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["synthetic_diagnostic"]
    assert digest(report["manifest"]) == report["manifest_sha256"]
    asset = Path(manifest["source_asset"])
    palette = asset.parent / (asset.stem + "_colored.colors.json")
    color = json.loads(
        (asset.parent / (asset.stem + "_colored.color_audit.json")).read_text()
    )
    assert color["passed"] and color["source_mjcf_sha256"] == digest(asset)
    legend = json.loads(palette.read_text())
    assert len(legend) == 30
    current = [episode(root, row) for row in report["rows"]]
    comparisons = []
    if a.before:
        previous = json.loads((a.before / "results.json").read_text())
        assert (
            previous["manifest_sha256"] == report["manifest_sha256"]
            and previous["seed"] == report["seed"]
            and previous["max_steps"] == report["max_steps"]
        )
        config_paths = [
            a.before.parent / "runtime/real_physics_ppo_diagnostic/config.json",
            root.parent / "runtime/real_physics_ppo_diagnostic/config.json",
        ]
        configs = [json.loads(path.read_text())["extra"] for path in config_paths]
        for key in [
            "tw_mjcf_sha256",
            "tw_physics_contract",
            "tw_joint_coordinate_contract",
            "tw_joint_speed_limits",
            "tw_state_joint_names",
            "tw_action_joint_names",
        ]:
            assert configs[0][key] == configs[1][key], key
        comparisons = [episode(a.before, row) for row in previous["rows"]]
    images = {row["agent"]: montage(root, row) for row in report["rows"]}
    sections = [
        '<!doctype html><meta charset="utf-8"><title>TW static control review</title>',
        "<style>body{font:16px system-ui;margin:24px;max-width:1300px;background:#f8f9fa}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #bbb}video{width:100%}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{max-width:100%}.note{padding:14px;background:#fff0c2}@media(max-width:800px){.views{grid-template-columns:1fr}}</style>",
        "<h1>TW static policy: final-checkpoint physical review</h1>",
        '<p class="note">Not promoted. A20s survival result does not prove flat-foot control, recovery, three training seeds or independent-motion success. Synthetic static reference; no HumanML3D skeleton source exists for this diagnostic.</p>',
        "<p>Each row uses its actual duration. Short clips must not be compared as20s drift. Contact polygon area uses positive native contact points projected into sole coordinates; it does not measure continuous pressure.</p>",
        "<table><tr><th>Run / agent</th><th>Actual seconds</th><th>RootXY max mm</th><th>Root tilt max deg</th><th>Body error mm</th><th>Sole tilt median L/R</th><th>Contact area ratio median L/R</th></tr>",
    ]
    for label, metrics in [("before", comparisons), ("current", current)]:
        for m in metrics:
            f = [m["contacts"][side]["last_up_to10s"] for side in ["left", "right"]]
            vals = [
                label + " / " + m["agent"],
                f"{m['seconds']:.2f}",
                f"{m['root_xy_max_mm']:.3f}",
                f"{m['root_tilt_max_deg']:.3f}",
                f"{m['mean_body_error_mm']:.3f}",
                " / ".join(f"{x['tilt_median_deg']:.3f}" for x in f),
                " / ".join(f"{x['contact_polygon_area_ratio_median']:.3f}" for x in f),
            ]
            sections.append(
                "<tr>"
                + "".join("<td>" + html.escape(v) + "</td>" for v in vals)
                + "</tr>"
            )
    sections.append("</table>")
    for row in report["rows"]:
        sections.append(
            f"<h2>{html.escape(row['agent'])}: {row['duration_s']:.2f}s; terminations {html.escape(str(row['termination_reasons']))}</h2><div class=views>"
        )
        for view in ["front", "side"]:
            sections.append(
                f'<section><h3>{view}</h3><video controls preload="metadata" src="{html.escape(row[view + "_video"])}"></video></section>'
            )
        sections.append(
            f'</div><img src="{images[row["agent"]]}" alt="Fixed start middle end samples">'
        )
    if a.before:
        link = os.path.relpath(a.before / "reviewed_index.html", root)
        sections.append(
            f'<p><a href="{html.escape(link)}">Preserved before-run videos</a></p>'
        )
    sections.append(
        "<p>Camera follows translation, with heading fixed at the initial world direction. Twelve fixed sampled frames are generated; their visual review is recorded separately. All four videos were fully decoded by the verifier.</p><h2>Joint-to-color legend</h2><p>Colors baked offline from saved precolored USD palette to a separate physics-identical MJCF; no runtime material binding.</p><table>"
    )
    for x in legend:
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", x["hex"])
        sections.append(
            f'<tr><td style="background:{x["hex"]}">{x["hex"]}</td><td>{html.escape(x["joint"])}</td><td>{html.escape(x["link"])}</td></tr>'
        )
    sections.append("</table>")
    (root / "reviewed_index.html").write_text("\n".join(sections))
    result = {
        "scope": "Actual episode metrics only; not acceptance or promotion",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_sha256": digest(__file__),
        "current": current,
        "before": comparisons,
        "visual_review": "Pending actual image inspection",
        "joint_color_entries": 30,
    }
    target.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "metrics": str(target),
                "page": str(root / "reviewed_index.html"),
                "episodes": len(current),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
