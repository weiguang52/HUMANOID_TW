"""CPU PD/contact diagnostic; this is not trained-policy evaluation."""

import argparse, importlib.util, json, math, xml.etree.ElementTree as E
from pathlib import Path
import mujoco, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--mjcf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
cp = (
    Path(__file__).resolve().parents[2]
    / "third_party/supermimic_mjlab/src/supermimic/config/tw/contract.py"
)
spec = importlib.util.spec_from_file_location("contract", cp)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
groups = [
    (c.LEG_MAJOR_JOINTS, 20, 1, 40, 0.001, 3),
    (c.FOOT_JOINTS, 10, 0.5, 15, 0.001, 3),
    (c.WAIST_JOINTS, 10, 0.5, 20, 0.001, 2),
    (c.ARM_JOINTS, 5, 0.25, 8, 0.0005, 2),
    (c.NECK_JOINTS, 3, 0.15, 3, 0.0002, 2),
]
r = E.parse(a.mjcf).getroot()
option = r.find("option")
option.set("integrator", "implicitfast")
option.set("timestep", ".005")
for g in r.findall(".//geom"):
    if "_collision" in (g.get("name") or ""):
        g.set("contype", "0")
        g.set("conaffinity", "1")
        g.set("friction", "1 .005 .0001")
E.SubElement(
    r.find("worldbody"),
    "geom",
    name="floor",
    type="plane",
    size="0 0 .05",
    contype="1",
    conaffinity="1",
    friction="1 .005 .0001",
)
act = E.SubElement(r, "actuator")
order = []
limits = []
velocity_limits = []
for names, kp, kd, effort, armature, speed in groups:
    for name in names:
        joint = next(j for j in r.findall(".//joint") if j.get("name") == name)
        joint.set("armature", str(armature))
        E.SubElement(
            act,
            "position",
            name=name + "_pd",
            joint=name,
            kp=str(kp),
            kv=str(kd),
            forcelimited="true",
            forcerange=f"{-effort} {effort}",
        )
        order.append(name)
        limits.append(effort)
        velocity_limits.append(speed)
m = mujoco.MjModel.from_xml_string(E.tostring(r, encoding="unicode"))
d = mujoco.MjData(m)
d.qpos[:3] = [0, 0, 0.264]
target = {n: 0.0 for n in order}
target["left_shoulder_roll_joint"] = -math.pi / 2
target["right_shoulder_roll_joint"] = math.pi / 2
for n, q in target.items():
    d.qpos[m.joint(n).qposadr[0]] = q
d.ctrl[:] = [target[n] for n in order]
vaddr = [m.joint(n).dofadr[0] for n in order]
torque_max = np.zeros(len(order))
speed_max = np.zeros(len(order))
contacts = 0
root = []
for step in range(4000):
    mujoco.mj_step(m, d)
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise ValueError("Nonfinite simulation")
    torque_max = np.maximum(torque_max, np.abs(d.actuator_force))
    speed_max = np.maximum(speed_max, np.abs(d.qvel[vaddr]))
    contacts += int(d.ncon > 0)
    if step % 4 == 0:
        root.append(d.qpos[:3].copy())
result = {
    "finite_20s": True,
    "torque_limits_pass": bool(np.all(torque_max <= np.array(limits) + 1e-6)),
    "physics_steps": 4000,
    "contact_steps": contacts,
    "root_start_m": root[0].tolist(),
    "root_end_m": root[-1].tolist(),
    "max_joint_speed_rad_s": float(speed_max.max()),
    "original_velocity_cap_exceeded_joints": [
        n for n, v, limit in zip(order, speed_max, velocity_limits) if v > limit + 1e-5
    ],
    "scope": "constant default-pose targets, CPU MuJoCo PD/contact diagnostic; not policy or mjlab runtime acceptance; velocity caps not yet enforced",
}
a.output.write_text(json.dumps(result, indent=2))
print(result)
