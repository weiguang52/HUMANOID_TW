"""Measure actual reference FK agreement for every frame; never alter motions."""

import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import mujoco

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--mjcf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
contract = (
    Path(__file__).resolve().parents[2]
    / "third_party/supermimic_mjlab/src/supermimic/config/tw/contract.py"
)
spec = importlib.util.spec_from_file_location("tw_contract", contract)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
model = mujoco.MjModel.from_xml_path(str(a.mjcf))
d = mujoco.MjData(model)
joint_names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
]
addresses = [model.joint(n).qposadr[0] for n in joint_names]
body_names = c.CUSTOM_HUMANOID_TRACKING_BODY_NAMES
body_ids = [model.body(n).id for n in body_names]
rows = []
for motion in json.load(a.manifest.open())["motions"]:
    with np.load(motion["file"], allow_pickle=False) as z:
        assert str(z["joint_coordinate_contract"][0]) == "tw123_yaw_waist_v3"
        nj = z["joint_names"].tolist()
        nb = z["body_names"].tolist()
        q = z["joint_pos"][:, [nj.index(n) for n in joint_names]]
        root = nb.index("base_link")
        reference = z["body_pos_w"][:, [nb.index(n) for n in body_names]]
        rq = z["body_quat_w"][:, [nb.index(n) for n in body_names]].astype(np.float64)
        base_pos = z["body_pos_w"][:, root].astype(np.float64)
        base_quat = z["body_quat_w"][:, root].astype(np.float64)
        base_quat /= np.linalg.norm(base_quat, axis=-1, keepdims=True)
        positions = []
        angles = []
        assert len(q) == motion["frames"] and np.isfinite(q).all()
        for i in range(len(q)):
            d.qpos[:] = model.qpos0
            d.qpos[:3] = base_pos[i]
            d.qpos[3:7] = base_quat[i]
            d.qpos[addresses] = q[i]
            mujoco.mj_kinematics(model, d)
            positions.extend(
                np.linalg.norm(d.xpos[body_ids] - reference[i], axis=-1).tolist()
            )
            norm = rq[i] / np.linalg.norm(rq[i], axis=-1, keepdims=True)
            dot = np.clip(np.abs(np.sum(d.xquat[body_ids] * norm, axis=-1)), 0, 1)
            angles.extend((2 * np.arccos(dot)).tolist())
        rows.append(
            {
                "id": motion["id"],
                "frames": len(q),
                "max_pos_m": max(positions),
                "p95_pos_m": float(np.percentile(positions, 95)),
                "max_ori_rad": max(angles),
                "p95_ori_rad": float(np.percentile(angles, 95)),
            }
        )
    if len(rows) % 25 == 0:
        print("motions", len(rows), flush=True)
result = {
    "status": "diagnostic_not_runtime_acceptance",
    "motions": len(rows),
    "frames": sum(x["frames"] for x in rows),
    "tracked_bodies": body_names,
    "max_position_error_m": max(x["max_pos_m"] for x in rows),
    "max_orientation_error_rad": max(x["max_ori_rad"] for x in rows),
    "rows": rows,
}
a.output.write_text(json.dumps(result, indent=2))
print({k: v for k, v in result.items() if k not in ["rows", "tracked_bodies"]})
