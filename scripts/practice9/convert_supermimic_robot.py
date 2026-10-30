"""Create a separate MuJoCo asset from the exact TW URDF; never edit the URDF."""

import argparse, json, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from stl_to_obj_exact import convert_stl

p = argparse.ArgumentParser()
p.add_argument("--urdf", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), "Refuse to overwrite a converted asset"
r = ET.parse(a.urdf).getroot()
m = r.find("mujoco")
if m is None:
    m = ET.SubElement(r, "mujoco")
c = m.find("compiler")
if c is None:
    c = ET.SubElement(m, "compiler")
c.set("fusestatic", "false")
c.set("discardvisual", "false")
c.set("balanceinertia", "false")
c.set("strippath", "false")
for link in r.findall("link"):
    for kind in ["collision", "visual"]:
        for index, geom in enumerate(link.findall(kind)):
            geom.set(
                "name",
                link.get("name")
                + "_"
                + kind
                + ("" if index == 0 else "_" + str(index)),
            )
mesh_records = {}
a.output.parent.mkdir(parents=True, exist_ok=True)
for mesh in r.findall(".//mesh"):
    f = Path(mesh.get("filename"))
    assert f.is_absolute() and f.is_file(), str(f)
    if f.suffix.lower() == ".stl":
        if str(f) not in mesh_records:
            mesh_records[str(f)] = convert_stl(
                f, a.output.parent / "meshes" / (f.stem + ".obj")
            )
        mesh.set("filename", mesh_records[str(f)]["obj"])
a.output.with_suffix(".meshes.json").write_text(
    json.dumps(list(mesh_records.values()), indent=2)
)
s = mujoco.MjSpec.from_string(ET.tostring(r, encoding="unicode"))
root = s.body("base_link")
assert root is not None
root.add_freejoint(name="floating_base")
s.option.timestep = 0.005
model = s.compile()
names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
]
expected = [j.get("name") for j in r.findall("joint") if j.get("type") != "fixed"]
assert len(names) == 30 and set(names) == set(expected), (
    len(names),
    set(expected) - set(names),
)
assert model.nq == 37 and model.nv == 36
masses = {
    link.get("name"): float(link.find("inertial/mass").get("value"))
    for link in r.findall("link")
    if link.find("inertial/mass") is not None
}
for name, mass in masses.items():
    assert np.isclose(model.body(name).mass[0], mass, rtol=1e-8, atol=1e-12), name
for j in r.findall("joint"):
    if j.get("type") == "fixed":
        continue
    limit = j.find("limit")
    i = model.joint(j.get("name")).id
    if j.get("type") != "continuous":
        assert np.allclose(
            model.jnt_range[i],
            [float(limit.get("lower")), float(limit.get("upper"))],
            atol=1e-10,
        )
a.output.parent.mkdir(parents=True, exist_ok=True)
s.to_file(str(a.output))
# MuJoCo 3.11 XML writer rounds values; restore full compiled body/inertial/joint precision.
xml = ET.parse(a.output)
fmt = lambda v: " ".join(format(float(x), ".17g") for x in np.asarray(v).reshape(-1))
for body in xml.getroot().findall(".//body"):
    b = model.body(body.get("name"))
    body.set("pos", fmt(b.pos))
    body.set("quat", fmt(b.quat))
    inert = body.find("inertial")
    if inert is not None:
        inert.set("pos", fmt(b.ipos))
        inert.set("quat", fmt(b.iquat))
        inert.set("mass", fmt(b.mass))
        inert.set("diaginertia", fmt(b.inertia))
        inert.attrib.pop("fullinertia", None)
for joint in xml.getroot().findall(".//joint"):
    if not joint.get("name"):
        continue
    j = model.joint(joint.get("name"))
    joint.set("pos", fmt(j.pos))
    joint.set("axis", fmt(j.axis))
    if bool(j.limited[0]):
        joint.set("range", fmt(j.range))
xml.write(a.output, encoding="unicode")
report = {
    "passed": True,
    "urdf_sha256": hashlib.sha256(a.urdf.read_bytes()).hexdigest(),
    "mujoco_version": mujoco.__version__,
    "joint_names_in_simulator_order": names,
    "nq": model.nq,
    "nv": model.nv,
    "mass_kg": float(model.body_mass.sum()),
    "scope": "compiled asset, joint set/limits and per-body masses; FK/inertia/contact/PD equivalence and runtime still require validation",
    "actuators_added": False,
}
a.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
