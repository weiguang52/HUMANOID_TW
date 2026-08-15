#!/usr/bin/env python3
"""Convert custom 30-DoF retarget outputs into validated practice-9 NPZ clips."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROCESS_TMP_DIR = Path(
    os.environ.get(
        "TMPDIR",
        f"{os.environ.get('GPUFREE_DATA_ROOT', '/root/gpufree-data')}/tmp/practice9_custom",
    )
)
PROCESS_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROCESS_TMP_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input-manifest", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--output-manifest", type=Path)
parser.add_argument(
    "--training-urdf",
    type=Path,
    default=Path(
        os.environ.get(
            "PRACTICE9_CUSTOM_URDF",
            "/root/gpufree-data/datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf",
        )
    ),
)
parser.add_argument("--ground-clearance", type=float, default=0.001)
parser.add_argument("--state-tolerance", type=float, default=1.0e-5)
parser.add_argument("--body-linear-velocity-p95-tolerance", type=float, default=0.05)
parser.add_argument("--body-angular-velocity-p95-tolerance", type=float, default=0.2)
parser.add_argument("--allow-quality-failures", action="store_true")
parser.add_argument("--continue-on-error", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
from unitree_rl_lab.assets.robots.custom_humanoid import (
    CUSTOM_HUMANOID_30DOF_CFG as ROBOT_CFG,
)
from unitree_rl_lab.assets.robots.custom_humanoid import (
    CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
)


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def load_link_collision_corners(
    urdf_path: Path, link_names: list[str]
) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    result: dict[str, np.ndarray] = {}
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    for name in link_names:
        corners = []
        for collision in links[name].findall("collision"):
            box = collision.find("geometry/box")
            if box is None:
                raise ValueError(f"{urdf_path}: {name} collision must be a box")
            size = np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
            origin = collision.find("origin")
            xyz = np.zeros(3) if origin is None else np.fromstring(
                origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64
            )
            rpy = np.zeros(3) if origin is None else np.fromstring(
                origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64
            )
            corners.append(xyz + (signs * (0.5 * size)) @ _rpy_matrix(rpy).T)
        if not corners:
            raise ValueError(f"{urdf_path}: {name} has no collision boxes")
        result[name] = np.concatenate(corners, axis=0)
    return result


def _quat_wxyz_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1.0e-8)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def compute_sole_heights(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: list[str],
    collision_corners: dict[str, np.ndarray],
) -> np.ndarray:
    heights = []
    for name, corners in collision_corners.items():
        body_index = body_names.index(name)
        rotation = _quat_wxyz_matrix(body_quat[:, body_index])
        corners_w = body_pos[:, body_index, None, :] + np.einsum(
            "tij,cj->tci", rotation, corners
        )
        heights.append(np.min(corners_w[..., 2], axis=1))
    return np.stack(heights, axis=1)


def so3_derivative(rotations_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
    if rotations_wxyz.shape[0] < 3:
        return torch.zeros((rotations_wxyz.shape[0], 3), device=rotations_wxyz.device)
    previous, following = rotations_wxyz[:-2], rotations_wxyz[2:]
    relative = quat_mul(following, quat_conjugate(previous))
    omega = axis_angle_from_quat(relative) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def load_retarget(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        stored_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_quat_xyzw = np.asarray(data["root_quat_xyzw"], dtype=np.float32)

    lookup = {name: index for index, name in enumerate(stored_names)}
    missing = [name for name in CUSTOM_HUMANOID_30DOF_JOINT_NAMES if name not in lookup]
    if missing:
        raise ValueError(f"{path}: missing joints {missing}")
    indexes = [lookup[name] for name in CUSTOM_HUMANOID_30DOF_JOINT_NAMES]
    joint_pos = joint_pos[:, indexes]

    length = joint_pos.shape[0]
    if length < 2 or root_pos.shape != (length, 3) or root_quat_xyzw.shape != (length, 4):
        raise ValueError(f"{path}: inconsistent retarget shapes")
    if not all(np.isfinite(value).all() for value in (joint_pos, root_pos, root_quat_xyzw)):
        raise ValueError(f"{path}: contains NaN or Inf")

    root_quat_xyzw /= np.linalg.norm(root_quat_xyzw, axis=-1, keepdims=True).clip(min=1.0e-8)
    for frame in range(1, length):
        if np.dot(root_quat_xyzw[frame - 1], root_quat_xyzw[frame]) < 0.0:
            root_quat_xyzw[frame] *= -1.0

    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    root_quat_t = torch.tensor(root_quat_wxyz, dtype=torch.float32, device=device)
    root_ang_vel = so3_derivative(root_quat_t, dt)

    return (
        fps,
        torch.tensor(joint_pos, dtype=torch.float32, device=device),
        torch.tensor(joint_vel, dtype=torch.float32, device=device),
        torch.tensor(root_pos, dtype=torch.float32, device=device),
        root_quat_t,
        torch.tensor(root_lin_vel, dtype=torch.float32, device=device),
        root_ang_vel,
    )


def convert_clip(
    sim: SimulationContext,
    scene: InteractiveScene,
    input_path: Path,
    output_path: Path,
    motion_id: str,
) -> dict[str, object]:
    robot = scene["robot"]
    joint_indexes, resolved_names = robot.find_joints(
        CUSTOM_HUMANOID_30DOF_JOINT_NAMES, preserve_order=True
    )
    if resolved_names != CUSTOM_HUMANOID_30DOF_JOINT_NAMES:
        raise ValueError(f"Simulator joint order mismatch: {resolved_names}")
    body_names = list(robot.body_names)

    (
        fps,
        joint_pos,
        joint_vel,
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
    ) = load_retarget(input_path, sim.device)

    log: dict[str, list[np.ndarray]] = {
        "joint_pos": [],
        "joint_vel": [],
        "root_state_w": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    for frame in range(joint_pos.shape[0]):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_quat[frame]
        root_state[:, 7:10] = root_lin_vel[frame]
        root_state[:, 10:13] = root_ang_vel[frame]
        robot.write_root_state_to_sim(root_state)

        robot.write_joint_state_to_sim(
            joint_pos[frame].unsqueeze(0),
            joint_vel[frame].unsqueeze(0),
            joint_ids=joint_indexes,
        )
        sim.render()
        scene.update(sim.get_physics_dt())
        actual_joint_pos = robot.data.joint_pos[0, joint_indexes].cpu().numpy().copy()
        actual_joint_vel = robot.data.joint_vel[0, joint_indexes].cpu().numpy().copy()
        actual_root_state = robot.data.root_state_w[0].cpu().numpy().copy()
        target_joint_pos = joint_pos[frame].cpu().numpy()
        target_joint_vel = joint_vel[frame].cpu().numpy()
        target_root_state = root_state[0].cpu().numpy()
        quat_error = min(
            np.linalg.norm(actual_root_state[3:7] - target_root_state[3:7]),
            np.linalg.norm(actual_root_state[3:7] + target_root_state[3:7]),
        )
        state_error = max(
            float(np.max(np.abs(actual_joint_pos - target_joint_pos))),
            float(np.max(np.abs(actual_joint_vel - target_joint_vel))),
            float(np.max(np.abs(actual_root_state[:3] - target_root_state[:3]))),
            float(quat_error),
            float(np.max(np.abs(actual_root_state[7:] - target_root_state[7:]))),
        )
        if state_error > args_cli.state_tolerance:
            raise ValueError(
                f"{motion_id}: simulator state readback error {state_error:.6g} "
                f"> {args_cli.state_tolerance:.6g} at frame {frame}"
            )
        log["joint_pos"].append(actual_joint_pos)
        log["joint_vel"].append(actual_joint_vel)
        log["root_state_w"].append(actual_root_state)
        for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            log[field].append(getattr(robot.data, field)[0].cpu().numpy().copy())

    arrays = {name: np.stack(values).astype(np.float32) for name, values in log.items()}
    root_state_np = arrays.pop("root_state_w")
    joint_pos_np = arrays.pop("joint_pos")
    joint_vel_np = arrays.pop("joint_vel")
    root_pos_np = root_state_np[:, :3].copy()
    root_pos_np[:, :2] -= scene.env_origins[0, :2].cpu().numpy()
    root_quat_np = root_state_np[:, 3:7]

    collision_corners = load_link_collision_corners(
        args_cli.training_urdf, ["left_foot", "right_foot"]
    )
    sole_z = compute_sole_heights(
        arrays["body_pos_w"], arrays["body_quat_w"], body_names, collision_corners
    )
    shift_z = args_cli.ground_clearance - float(np.min(sole_z))
    root_pos_np[:, 2] += shift_z
    arrays["body_pos_w"][:, :, 2] += shift_z
    sole_z += shift_z

    base_index = body_names.index("base_link")
    base_pos_error = float(
        np.max(np.abs(arrays["body_pos_w"][:, base_index] - root_pos_np))
    )
    base_quat_error = float(
        np.max(
            np.minimum(
                np.linalg.norm(arrays["body_quat_w"][:, base_index] - root_quat_np, axis=1),
                np.linalg.norm(arrays["body_quat_w"][:, base_index] + root_quat_np, axis=1),
            )
        )
    )
    if max(base_pos_error, base_quat_error) > args_cli.state_tolerance:
        raise ValueError(
            f"{motion_id}: base/root FK mismatch pos={base_pos_error:.6g}, "
            f"quat={base_quat_error:.6g}"
        )

    dt = 1.0 / fps
    body_lin_fd = np.gradient(arrays["body_pos_w"], dt, axis=0)
    lin_velocity_error_p95 = float(
        np.quantile(np.linalg.norm(body_lin_fd - arrays["body_lin_vel_w"], axis=-1), 0.95)
    )
    body_quat_contiguous = arrays["body_quat_w"].copy()
    for frame in range(1, len(body_quat_contiguous)):
        flip = np.sum(body_quat_contiguous[frame - 1] * body_quat_contiguous[frame], axis=-1) < 0
        body_quat_contiguous[frame, flip] *= -1
    quat_tensor = torch.tensor(body_quat_contiguous, device=sim.device)
    body_ang_fd = torch.stack(
        [so3_derivative(quat_tensor[:, index], dt) for index in range(len(body_names))],
        dim=1,
    ).cpu().numpy()
    ang_velocity_error_p95 = float(
        np.quantile(np.linalg.norm(body_ang_fd - arrays["body_ang_vel_w"], axis=-1), 0.95)
    )
    fk_quality_pass = (
        lin_velocity_error_p95 <= args_cli.body_linear_velocity_p95_tolerance
        and ang_velocity_error_p95 <= args_cli.body_angular_velocity_p95_tolerance
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([1], dtype=np.int32),
        fps=np.asarray([fps], dtype=np.float32),
        motion_id=np.asarray([motion_id]),
        joint_names=np.asarray(CUSTOM_HUMANOID_30DOF_JOINT_NAMES),
        body_names=np.asarray(body_names),
        joint_pos=joint_pos_np,
        joint_vel=joint_vel_np,
        root_pos=root_pos_np,
        root_quat_wxyz=root_quat_np,
        **arrays,
    )
    return {
        "id": motion_id,
        "file": str(output_path.resolve()),
        "weight": 1.0,
        "frames": int(joint_pos.shape[0]),
        "ground_shift_z": shift_z,
        "sole_min_z": float(np.min(sole_z)),
        "sole_p05_z": float(np.quantile(sole_z, 0.05)),
        "base_pos_error_max": base_pos_error,
        "base_quat_error_max": base_quat_error,
        "body_linear_velocity_error_p95": lin_velocity_error_p95,
        "body_angular_velocity_error_p95": ang_velocity_error_p95,
        "fk_quality_pass": fk_quality_pass,
        "source_file": str(input_path.resolve()),
    }


def main() -> None:
    payload = json.loads(args_cli.input_manifest.read_text(encoding="utf-8"))
    motions = payload.get("motions", [])
    if not motions:
        raise ValueError(f"No motions in {args_cli.input_manifest}")
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args_cli.output_manifest or args_cli.output_dir / "manifest.json"

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(payload.get("target_fps", 50))
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()

    converted = []
    failures = []
    for index, item in enumerate(motions, start=1):
        motion_id = str(item["id"])
        input_path = Path(item["file"]).expanduser()
        if not input_path.is_absolute():
            input_path = args_cli.input_manifest.parent / input_path
        input_path = input_path.resolve()
        source_quality_pass = item.get("quality_pass") is True
        output_path = args_cli.output_dir / f"{motion_id}.npz"
        try:
            if not source_quality_pass and not args_cli.allow_quality_failures:
                raise ValueError(
                    f"{motion_id}: retarget record is not marked quality_pass=true"
                )
            print(f"[{index}/{len(motions)}] FK {motion_id}", flush=True)
            record = convert_clip(sim, scene, input_path, output_path, motion_id)
            record["weight"] = float(item.get("weight", 1.0))
            record["retarget_quality"] = item.get("quality", {})
            record["quality_pass"] = bool(source_quality_pass and record["fk_quality_pass"])
            if not record["quality_pass"] and not args_cli.allow_quality_failures:
                output_path.unlink(missing_ok=True)
                raise ValueError(f"{motion_id}: FK or retarget quality gate failed")
            converted.append(record)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            failures.append({"id": motion_id, "file": str(input_path), "error": str(exc)})
            traceback.print_exc()
            if not args_cli.continue_on_error:
                raise

    result = {
        "schema_version": 1,
        "target_fps": float(payload.get("target_fps", 50)),
        "robot": "urdf0711_training_30dof",
        "joint_names": CUSTOM_HUMANOID_30DOF_JOINT_NAMES,
        "motions": converted,
        "failures": failures,
    }
    output_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(converted)} clips and {len(failures)} failures to {output_manifest}"
    )
    if not converted:
        raise RuntimeError("No clips passed FK conversion")


if __name__ == "__main__":
    try:
        main()
    finally:
        # This is a non-interactive batch converter with no Replicator work.
        # Full Kit cleanup can spin indefinitely after all validated outputs
        # are closed; the documented immediate shutdown releases the GPU.
        simulation_app.close(
            wait_for_replicator=False, skip_cleanup=True
        )
