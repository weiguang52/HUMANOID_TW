"""Verify encoded frame counts and build an honest fixed-manifest replay index."""

import argparse
import html
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--palette", type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.evaluation / "results.json").read_text())
    source = {r["id"]: r for r in json.loads(a.source_index.read_text())}
    palette = json.loads(a.palette.read_text())
    audits = []
    rows = []
    for row in report["rows"]:
        assert row["id"] in source
        links = []
        for view in ("front", "side"):
            path = a.evaluation / row[view + "_video"]
            stream = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,nb_frames,r_frame_rate,width,height,pix_fmt",
                        "-of",
                        "json",
                        str(path),
                    ],
                    text=True,
                )
            )["streams"][0]
            assert stream["codec_name"] == "h264"
            assert int(stream["nb_frames"]) == row["steps"]
            num, den = map(float, stream["r_frame_rate"].split("/"))
            assert abs(num / den - 50) < 1e-6
            assert (int(stream["width"]), int(stream["height"])) == (960, 640)
            audits.append(
                dict(
                    id=row["id"],
                    agent=row["agent"],
                    view=view,
                    video=str(path),
                    **stream,
                )
            )
            links.append(f'<a href="{html.escape(path.name)}">{view}</a>')
        original = a.source_index.parent / source[row["id"]]["video"]
        assert original.is_file()
        links.append(
            f'<a href="{html.escape(os.path.relpath(original, a.evaluation))}">original HumanML3D</a>'
        )
        state = (
            "FAILED: " + ",".join(row["termination_reasons"])
            if row["terminated"]
            else (
                "requested window completed"
                if row["completed_requested_window"]
                else "truncated"
            )
        )
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{v}</td>"
                for v in [
                    html.escape(row["id"]),
                    row["agent"],
                    f"{row['duration_s']:.2f} / {row['planned_duration_s']:.2f}",
                    html.escape(state),
                    f"{row['mean_anchor_error_m'] * 1000:.1f}",
                    f"{row['mean_body_error_m'] * 1000:.1f}",
                    " | ".join(links),
                ]
            )
            + "</tr>"
        )
    body = """<!doctype html><meta charset="utf-8"><title>TW SuperMimic evaluation</title>
<style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#162230}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd4dc;padding:8px;text-align:left}code{overflow-wrap:anywhere}a{color:#086ac2}</style>
<h1>TW SuperMimic: fixed-manifest evaluation</h1>
<p>Single seed, one deterministic start per motion and agent. No restart after failure. No best-checkpoint selection. This is a method-transfer experiment, not proof of a production-ready policy.</p>
<p>Robot replays: 50 fps, colored asset baked before simulation. Front starts on robot +X; side on initial -Y. Cameras follow translation and retain initial world heading during turns. Colors were transferred offline from the existing precolored USD palette to MuJoCo MJCF, with all non-color compiled arrays unchanged.</p>
<p>Original HumanML3D videos use original 20 fps and fixed cameras; robot references may be time-stretched. Videos are not synchronized. Errors are averaged over actual episode duration only; early failures cannot be compared as 20-second drift. MuJoCo results cannot be directly treated as PhysX comparisons.</p>"""
    body += f"<p>Checkpoint SHA256: <code>{html.escape(report['checkpoint_sha256'])}</code><br>Manifest SHA256: <code>{html.escape(report['manifest_sha256'])}</code></p>"
    body += (
        "<table><tr><th>Clip</th><th>Agent</th><th>Actual / requested seconds</th><th>Outcome</th><th>Mean anchor error mm</th><th>Mean body error mm</th><th>Videos</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    body += "<h2>Joint color legend</h2><table><tr><th>Joint</th><th>Child link</th><th>Color</th></tr>"
    for r in palette:
        body += f'<tr><td>{html.escape(r["joint"])}</td><td>{html.escape(r["link"])}</td><td style="background:{html.escape(r["hex"])}">{html.escape(r["hex"])}</td></tr>'
    body += '</table><p><a href="results.json">Numerical results</a> | <a href="video_audit.json">Encoding/frame audit</a></p>'
    (a.evaluation / "index.html").write_text(body)
    (a.evaluation / "video_audit.json").write_text(
        json.dumps(
            {
                "passed": True,
                "videos": audits,
                "scope": "Encoding/frame verification only; visual framing, attachment and behavior require separate inspection.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Verified", len(audits), "videos; visual acceptance remains separate")


if __name__ == "__main__":
    main()
