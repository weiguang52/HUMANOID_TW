#!/usr/bin/env python3
"""Generate a conservative 30-DoF training URDF from urdf0711.

The source CAD export is never modified. The generated file converts controlled
joints to limited revolute joints, fixes the two extra hand-roll joints, keeps
both foot-roll joints active, removes invalid foot-end payloads, replaces foot
mesh collisions with stable boxes, and resolves meshes on the data disk.

The limits are bootstrap values derived from the old URDF and retargeting
configuration. Replace them with measured limits before sim-to-real deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_ROOT = Path(os.environ.get("GPUFREE_DATA_ROOT", "/root/gpufree-data"))
DEFAULT_SOURCE = DATA_ROOT / "projects/urdf0711/urdf/urdf0711.urdf"
DEFAULT_MESH_ROOT = DATA_ROOT / "projects/urdf0711/meshes"
DEFAULT_OUTPUT = DATA_ROOT / "datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf"

# lower [rad], upper [rad], effort [Nm], velocity [rad/s]
JOINT_SPECS: dict[str, tuple[float, float, float, float]] = {
    "left_hip_linkage_pitch": (-1.57, 1.57, 100.0, 3.0),
    "left_thigh_roll": (-0.79, 0.79, 100.0, 3.0),
    "left_knee_linkage_yaw": (-1.05, 1.05, 80.0, 3.0),
    "left_mid_leg_pitch": (0.0, 2.09, 80.0, 3.0),
    "left_calf_yaw": (-0.87, 0.87, 60.0, 3.0),
    "left_ankle_pitch": (-0.79, 0.79, 60.0, 3.0),
    "left_foot_roll": (-0.35, 0.35, 60.0, 3.0),
    "right_hip_linkage_pitch": (-1.57, 1.57, 100.0, 3.0),
    "right_thigh_roll": (-0.79, 0.79, 100.0, 3.0),
    "right_knee_linkage_yaw": (-1.05, 1.05, 80.0, 3.0),
    "right_mid_leg_pitch": (-2.09, 0.0, 80.0, 3.0),
    "right_calf_yaw": (-0.87, 0.87, 60.0, 3.0),
    "right_ankle_pitch": (-0.79, 0.79, 60.0, 3.0),
    "right_foot_roll": (-0.35, 0.35, 60.0, 3.0),
    "left_shoulder_linkage_pitch": (-2.09, 3.14, 30.0, 2.0),
    "left_upper_arm_roll": (-1.57, 1.57, 30.0, 2.0),
    "left_elbow_linkage_pitch": (-1.05, 1.05, 25.0, 2.0),
    "left_force_arm_yaw": (-2.62, 0.0, 25.0, 2.0),
    "left_wrist_pitch": (-1.22, 1.22, 20.0, 2.0),
    "right_shoulder_linkage_pitch": (-3.14, 2.09, 30.0, 2.0),
    "right_upper_arm_roll": (-1.57, 1.57, 30.0, 2.0),
    "right_elbow_linkage_pitch": (-1.05, 1.05, 25.0, 2.0),
    "right_force_arm_yaw": (0.0, 2.62, 25.0, 2.0),
    "right_wrist_pitch": (-1.22, 1.22, 20.0, 2.0),
    "waist_yaw": (-1.57, 1.57, 50.0, 2.0),
    "gearbox_roll": (-0.52, 0.52, 50.0, 2.0),
    "chest_pitch": (-0.79, 0.79, 50.0, 2.0),
    "neck": (-1.05, 1.05, 20.0, 2.0),
    "neck_linkage_roll": (-0.52, 0.52, 15.0, 2.0),
    "head_pitch": (-0.79, 0.79, 15.0, 2.0),
}

FIXED_HAND_JOINTS = {"left_hand_roll", "right_hand_roll"}
EMPTY_FOOT_END_LINKS = {"left_foot_end", "right_foot_end"}


def _binary_stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Invalid STL file: {path}")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if triangle_count == 0 or len(raw) != 84 + triangle_count * 50:
        raise ValueError(f"Expected a non-empty binary STL: {path}")
    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    for triangle in range(triangle_count):
        offset = 84 + triangle * 50 + 12
        for vertex in range(3):
            xyz = struct.unpack_from("<fff", raw, offset + vertex * 12)
            for axis, value in enumerate(xyz):
                lower[axis] = min(lower[axis], value)
                upper[axis] = max(upper[axis], value)
    return lower, upper


def _replace_foot_collision(link: ET.Element, mesh_root: Path) -> None:
    mesh_path = mesh_root / f"{link.attrib['name']}.STL"
    lower, upper = _binary_stl_bounds(mesh_path)
    size = [max(upper[i] - lower[i], 1.0e-4) for i in range(3)]
    center = [(upper[i] + lower[i]) * 0.5 for i in range(3)]
    for collision in list(link.findall("collision")):
        link.remove(collision)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(
        collision,
        "origin",
        xyz=" ".join(f"{value:.9g}" for value in center),
        rpy="0 0 0",
    )
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size=" ".join(f"{value:.9g}" for value in size))


def generate(source: Path, mesh_root: Path, output: Path) -> dict[str, object]:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "urdf0711_training_30dof")

    seen: set[str] = set()
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        if name in FIXED_HAND_JOINTS:
            joint.set("type", "fixed")
            for tag in ("limit", "dynamics"):
                element = joint.find(tag)
                if element is not None:
                    joint.remove(element)
            continue
        if name not in JOINT_SPECS:
            if joint.attrib.get("type") != "fixed":
                raise ValueError(f"Unclassified active joint: {name}")
            continue
        seen.add(name)
        lower, upper, effort, velocity = JOINT_SPECS[name]
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.attrib.update(
            lower=f"{lower:.8g}",
            upper=f"{upper:.8g}",
            effort=f"{effort:.8g}",
            velocity=f"{velocity:.8g}",
        )
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.attrib.update(damping="0", friction="0")

    missing = set(JOINT_SPECS) - seen
    if missing:
        raise ValueError(f"Missing expected joints: {sorted(missing)}")

    for link in root.findall("link"):
        name = link.attrib["name"]
        if name in EMPTY_FOOT_END_LINKS:
            for tag in ("visual", "collision", "inertial"):
                for element in list(link.findall(tag)):
                    link.remove(element)
            continue
        if name in {"left_foot", "right_foot"}:
            _replace_foot_collision(link, mesh_root)
        for mesh in link.findall(".//mesh"):
            basename = Path(mesh.attrib["filename"]).name
            resolved = (mesh_root / basename).resolve()
            if not resolved.is_file() or resolved.stat().st_size <= 84:
                raise ValueError(f"Missing or empty mesh referenced by {name}: {resolved}")
            mesh.set("filename", str(resolved))

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)

    metadata = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_root": str(mesh_root.resolve()),
        "output": str(output.resolve()),
        "active_joint_count": len(JOINT_SPECS),
        "active_joint_names": list(JOINT_SPECS),
        "fixed_hand_joints": sorted(FIXED_HAND_JOINTS),
        "removed_invalid_foot_end_payloads": sorted(EMPTY_FOOT_END_LINKS),
        "limits_are_bootstrap_only": True,
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.source, args.mesh_root, args.output), indent=2))


if __name__ == "__main__":
    main()
