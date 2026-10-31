import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

root = Path("/root/gpufree-data/datasets/practice9")
p = root / "training_speed_blend_20261026/train.json"
manifest = json.loads(p.read_text())
m = mujoco.MjModel.from_xml_path(str(root / "supermimic_tw_20261031/robot/tw30.xml"))


def rotate(q, v):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    t = 2 * np.cross(q[:, None, 1:], v[None, :, :])
    return v[None, :, :] + q[:, None, :1] * t + np.cross(q[:, None, 1:], t)


rows = []
heldout = {"003245", "012584", "003363", "008312", "005048", "004518", "010195"}
assert not heldout.intersection(x["id"] for x in manifest["motions"])
data = mujoco.MjData(m)
for motion in manifest["motions"]:
    d = np.load(motion["file"])
    names = d["body_names"].tolist()
    feet = {}
    fk_error = 0.0
    for t in sorted(
        set([0, len(d["joint_pos"]) - 1] + list(range(0, len(d["joint_pos"]), 100)))
    ):
        data.qpos[:3] = d["root_pos"][t]
        data.qpos[3:7] = d["root_quat_wxyz"][t]
        for j, name in enumerate(d["joint_names"].tolist()):
            data.qpos[int(m.joint(name).qposadr[0])] = d["joint_pos"][t, j]
        mujoco.mj_forward(m, data)
        for side in ["left", "right"]:
            idx = names.index(side + "_foot")
            bid = m.body(side + "_foot").id
            fk_error = max(
                fk_error,
                float(np.max(np.abs(data.xpos[bid] - d["body_pos_w"][t, idx]))),
            )
    assert fk_error < 1e-5, (motion["id"], fk_error)
    for side in ["left", "right"]:
        g = m.geom(side + "_foot_collision")
        idx = names.index(side + "_foot")
        assert np.allclose(g.quat, [1, 0, 0, 0])
        corners = np.array(list(itertools.product([-1, 1], repeat=3))) * g.size + g.pos
        q = d["body_quat_w"][:, idx]
        pos = d["body_pos_w"][:, idx]
        z = (rotate(q, corners) + pos[:, None, :])[:, :, 2].min(axis=1)
        normals = rotate(q, np.array([[0.0, 0.0, 1.0]]))[:, 0]
        tilt = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
        near = z <= 0.002
        fps = float(d["fps"].item())
        speed = np.linalg.norm(np.gradient(pos.astype(float), 1 / fps, axis=0), axis=1)
        omega = np.linalg.norm(d["body_ang_vel_w"][:, idx], axis=1)
        quiet = near & (speed <= 0.03) & (omega <= 0.5)
        edge = quiet & (tilt > 15)
        changes = np.diff(np.r_[False, edge, False].astype(int))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        runs = [
            {"start": int(a), "end_exclusive": int(b), "seconds": float((b - a) / fps)}
            for a, b in zip(starts, ends, strict=True)
        ]
        feet[side] = {
            "quiet_near_ground_frames": int(quiet.sum()),
            "quiet_tilt_p95_deg": float(np.percentile(tilt[quiet], 95))
            if quiet.any()
            else None,
            "quiet_tilt_over15_runs": runs,
            "quiet_tilt_over15_longest_s": max(
                (v["seconds"] for v in runs), default=0.0
            ),
            "minimum_mm": float(z.min() * 1000),
            "maximum_mm": float(z.max() * 1000),
            "p95_mm": float(np.percentile(z, 95) * 1000),
            "fraction_above_5mm": float(np.mean(z > 0.005)),
            "fraction_above_10mm": float(np.mean(z > 0.010)),
            "near_ground_frames": int(near.sum()),
            "near_ground_tilt_p95_deg": float(np.percentile(tilt[near], 95))
            if near.any()
            else None,
        }
    rows.append(
        {
            "id": motion["id"],
            "file": motion["file"],
            "frames": len(pos),
            "source_family": motion.get("source_family"),
            "sampled_foot_fk_max_error_m": fk_error,
            "feet": feet,
        }
    )
out = (
    root
    / "supermimic_tw_20261031/longterm_diagnostics/training_sole_support_audit_v2.json"
)
result = {
    "manifest": str(p),
    "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "count": len(rows),
    "test_id_overlap": [],
    "model_sha256": hashlib.sha256(
        (root / "supermimic_tw_20261031/robot/tw30.xml").read_bytes()
    ).hexdigest(),
    "quiet_filter_description": "Descriptive only: corner min z <=2mm, body-origin finite-difference speed <=0.03m/s, angular speed <=0.5rad/s. Tilt>15deg contiguous durations recorded. Not force-based contacts or an acceptance gate.",
    "scope": "All 332 training clips. Descriptive 2/5/10mm bins are not acceptance gates; no test selection, no training started. Reference kinematics do not prove dynamically feasible support.",
    "motions": rows,
}
out.write_text(json.dumps(result, indent=2))
print("AUDIT", out, "count", len(rows))
print(
    "both_feet_never_above_5mm",
    sum(all(x["feet"][s]["maximum_mm"] <= 5 for s in ["left", "right"]) for x in rows),
)
print(
    "both_feet_have_above_10mm",
    sum(all(x["feet"][s]["maximum_mm"] > 10 for s in ["left", "right"]) for x in rows),
)
