"""Compare MuJoCo compiled kinematics/inertias against independent URDF algebra."""

import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

p = argparse.ArgumentParser()
p.add_argument("--urdf", type=Path, required=True)
p.add_argument("--mjcf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
urdf = ET.parse(a.urdf).getroot()
model = mujoco.MjModel.from_xml_path(str(a.mjcf))
data = mujoco.MjData(model)


def xyz(s):
    return np.fromstring(s or "0 0 0", sep=" ")


def rot(axis, q):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


def origin(e):
    T = np.eye(4)
    if e is not None:
        x, y, z = xyz(e.get("rpy"))
        T[:3, :3] = rot([0, 0, 1], z) @ rot([0, 1, 0], y) @ rot([1, 0, 0], x)
        T[:3, 3] = xyz(e.get("xyz"))
    return T


inertia_error = 0.0
com_error = 0.0
for link in urdf.findall("link"):
    inert = link.find("inertial")
    if inert is None:
        continue
    body = model.body(link.get("name"))
    T = origin(inert.find("origin"))
    v = inert.find("inertia").attrib
    I = np.array(
        [
            [float(v["ixx"]), float(v["ixy"]), float(v["ixz"])],
            [float(v["ixy"]), float(v["iyy"]), float(v["iyz"])],
            [float(v["ixz"]), float(v["iyz"]), float(v["izz"])],
        ]
    )
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, body.iquat)
    R = R.reshape(3, 3)
    inertia_error = max(
        inertia_error,
        float(
            np.max(
                np.abs(R @ np.diag(body.inertia) @ R.T - T[:3, :3] @ I @ T[:3, :3].T)
            )
        ),
    )
    com_error = max(com_error, float(np.max(np.abs(body.ipos - T[:3, 3]))))
rng = np.random.default_rng(42)
pos_error = 0.0
rotation_error = 0.0
links = urdf.findall("link")
joints = urdf.findall("joint")
for sample in range(50):
    q = {}
    for j in joints:
        if j.get("type") == "fixed":
            continue
        limit = j.find("limit")
        lo = float(limit.get("lower", -0.5))
        hi = float(limit.get("upper", 0.5))
        q[j.get("name")] = float(rng.uniform(lo, hi))
    data.qpos[:] = model.qpos0
    data.qpos[:3] = [0.03, -0.04, 0.264]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in q.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    T = np.eye(4)
    T[:3, 3] = data.qpos[:3]
    poses = {"base_link": T}
    pending = list(joints)
    while pending:
        previous = len(pending)
        for j in pending[:]:
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            if parent not in poses:
                continue
            local = origin(j.find("origin"))
            if j.get("type") != "fixed":
                delta = np.eye(4)
                axis = xyz(j.find("axis").get("xyz"))
                if j.get("type") in ("revolute", "continuous"):
                    delta[:3, :3] = rot(axis, q[j.get("name")])
                elif j.get("type") == "prismatic":
                    delta[:3, 3] = axis * q[j.get("name")]
                else:
                    raise ValueError(j.get("type"))
                local = local @ delta
            poses[child] = poses[parent] @ local
            pending.remove(j)
        assert len(pending) < previous, "Unresolved URDF tree"
    for name, T in poses.items():
        i = model.body(name).id
        pos_error = max(pos_error, float(np.max(np.abs(data.xpos[i] - T[:3, 3]))))
        rotation_error = max(
            rotation_error,
            float(np.max(np.abs(data.xmat[i].reshape(3, 3) - T[:3, :3]))),
        )
assert len(poses) == len(links)
result = {
    "passed": pos_error < 1e-8
    and rotation_error < 1e-8
    and inertia_error < 1e-10
    and com_error < 1e-10,
    "poses": 50,
    "links": len(links),
    "seed": 42,
    "max_position_error_m": pos_error,
    "max_rotation_matrix_error": rotation_error,
    "max_inertia_tensor_error": inertia_error,
    "max_inertial_com_error_m": com_error,
    "scope": "kinematics and inertias, not contact/actuator dynamics or policy quality",
}
a.output.write_text(json.dumps(result, indent=2))
print(result)
assert result["passed"]
