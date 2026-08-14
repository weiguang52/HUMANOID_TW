#!/usr/bin/env python3
"""HumanML3D -> existing IK_V2 -> custom 30-DoF bootstrap retargeting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/root/gpufree-data")
DEFAULT_IK_ROOT = DATA_ROOT / "projects/IK_V2"
DEFAULT_TRAINING_URDF = DATA_ROOT / "datasets/practice9/custom_robot/urdf/urdf0711_training_30dof.urdf"
DEFAULT_INPUT = DATA_ROOT / "datasets/HumanML3D-official/HumanML3D/new_joints"
DEFAULT_OUTPUT = DATA_ROOT / "datasets/practice9/humanml3d_custom30/retargeted"

CUSTOM_JOINT_NAMES = [
    "left_hip_linkage_pitch", "left_thigh_roll", "left_knee_linkage_yaw",
    "left_mid_leg_pitch", "left_calf_yaw", "left_ankle_pitch", "left_foot_roll",
    "right_hip_linkage_pitch", "right_thigh_roll", "right_knee_linkage_yaw",
    "right_mid_leg_pitch", "right_calf_yaw", "right_ankle_pitch", "right_foot_roll",
    "waist_yaw", "gearbox_roll", "chest_pitch",
    "left_shoulder_linkage_pitch", "left_upper_arm_roll", "left_elbow_linkage_pitch",
    "left_force_arm_yaw", "left_wrist_pitch",
    "right_shoulder_linkage_pitch", "right_upper_arm_roll", "right_elbow_linkage_pitch",
    "right_force_arm_yaw", "right_wrist_pitch",
    "neck", "neck_linkage_roll", "head_pitch",
]

# new_name: (old_name, sign, zero offset)
MAPPING = {
    "left_hip_linkage_pitch": ("left_hip_pitch_joint", -1.0, 0.0),
    "left_thigh_roll": ("left_hip_roll_joint", 1.0, 0.0),
    "left_knee_linkage_yaw": ("left_hip_yaw_joint", 1.0, 0.0),
    "left_mid_leg_pitch": ("left_knee_pitch_joint", -1.0, 0.0),
    "left_calf_yaw": ("left_ankle_yaw_joint", -1.0, 0.0),
    "left_ankle_pitch": ("left_ankle_pitch_joint", 1.0, 0.0),
    "left_foot_roll": (None, 0.0, 0.0),
    "right_hip_linkage_pitch": ("right_hip_pitch_joint", -1.0, 0.0),
    "right_thigh_roll": ("right_hip_roll_joint", 1.0, 0.0),
    "right_knee_linkage_yaw": ("right_hip_yaw_joint", 1.0, 0.0),
    "right_mid_leg_pitch": ("right_knee_pitch_joint", 1.0, 0.0),
    "right_calf_yaw": ("right_ankle_yaw_joint", 1.0, 0.0),
    "right_ankle_pitch": ("right_ankle_pitch_joint", 1.0, 0.0),
    "right_foot_roll": (None, 0.0, 0.0),
    "waist_yaw": ("waist_yaw_joint", 1.0, 0.0),
    "gearbox_roll": ("waist_roll_joint", 1.0, 0.0),
    "chest_pitch": ("waist_pitch_joint", -1.0, 0.0),
    "left_shoulder_linkage_pitch": ("left_shoulder_pitch_joint", -1.0, 0.0),
    "left_upper_arm_roll": ("left_shoulder_roll_joint", 1.0, math.pi / 2.0),
    "left_elbow_linkage_pitch": ("left_shoulder_yaw_joint", 1.0, 0.0),
    "left_force_arm_yaw": ("left_elbow_pitch_joint", 1.0, 0.0),
    "left_wrist_pitch": ("left_wrist_yaw_joint", -1.0, 0.0),
    "right_shoulder_linkage_pitch": ("right_shoulder_pitch_joint", -1.0, 0.0),
    "right_upper_arm_roll": ("right_shoulder_roll_joint", 1.0, -math.pi / 2.0),
    "right_elbow_linkage_pitch": ("right_shoulder_yaw_joint", 1.0, 0.0),
    "right_force_arm_yaw": ("right_elbow_pitch_joint", 1.0, 0.0),
    "right_wrist_pitch": ("right_wrist_yaw_joint", -1.0, 0.0),
    "neck": ("neck_yaw_joint", 1.0, 0.0),
    "neck_linkage_roll": ("neck_roll_joint", 1.0, 0.0),
    "head_pitch": ("neck_pitch_joint", -1.0, 0.0),
}


def load_ik_module(ik_root: Path):
    module_path = ik_root / "ik_redirection_npy.py"
    spec = importlib.util.spec_from_file_location("ik_v2_redirection", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ik_root))
    spec.loader.exec_module(module)
    return module


def load_limits(training_urdf: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(training_urdf).getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    lower, upper, velocity = [], [], []
    for name in CUSTOM_JOINT_NAMES:
        limit = by_name[name].find("limit")
        if limit is None:
            raise ValueError(f"Missing limit for {name}")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
    return np.asarray(lower), np.asarray(upper), np.asarray(velocity)


def validate_humanml3d(joints: np.ndarray, path: Path) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise ValueError(f"{path}: expected [T,22,3], got {joints.shape}")
    if joints.shape[0] < 9:
        raise ValueError(f"{path}: at least 9 frames are required, got {joints.shape[0]}")
    if not np.isfinite(joints).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return joints


def map_28_to_30(old_dof: np.ndarray, old_names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(old_names)}
    output = np.zeros((old_dof.shape[0], len(CUSTOM_JOINT_NAMES)), dtype=np.float64)
    for target_index, new_name in enumerate(CUSTOM_JOINT_NAMES):
        old_name, sign, offset = MAPPING[new_name]
        if old_name is not None:
            output[:, target_index] = sign * old_dof[:, lookup[old_name]] + offset
    return output


def restore_root_trajectory(
    joints: np.ndarray, output_frames: int, root_scale: float, nominal_height: float
) -> tuple[np.ndarray, np.ndarray]:
    # Same coordinates as IK_V2: HumanML3D [x,y,z] -> robot [z,x,y].
    world = joints[..., [2, 0, 1]]
    pelvis = world[:, 0]
    lateral = world[:, 2] - world[:, 1]  # right hip - left hip
    heading = np.unwrap(np.arctan2(lateral[:, 1], lateral[:, 0]))
    heading_delta = heading - heading[0]

    # Normalize the initial lateral direction to robot -Y, retain subsequent turns.
    align = -math.pi / 2.0 - heading[0]
    c, s = math.cos(align), math.sin(align)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    relative = pelvis - pelvis[0]
    relative_xy = relative[:, :2] @ rotation.T

    source_t = np.linspace(0.0, 1.0, joints.shape[0])
    output_t = np.linspace(0.0, 1.0, output_frames)
    root_pos = np.empty((output_frames, 3), dtype=np.float64)
    root_pos[:, 0] = np.interp(output_t, source_t, relative_xy[:, 0] * root_scale)
    root_pos[:, 1] = np.interp(output_t, source_t, relative_xy[:, 1] * root_scale)
    root_pos[:, 2] = nominal_height + np.interp(output_t, source_t, relative[:, 2] * root_scale)
    yaw = np.interp(output_t, source_t, heading_delta)

    root_quat_xyzw = np.zeros((output_frames, 4), dtype=np.float64)
    root_quat_xyzw[:, 2] = np.sin(0.5 * yaw)
    root_quat_xyzw[:, 3] = np.cos(0.5 * yaw)
    return root_pos, root_quat_xyzw


def _gradient(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=2)


def _yaw_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.unwrap(2.0 * np.arctan2(quat[:, 2], quat[:, 3]))


def _quat_xyzw_from_yaw(yaw: np.ndarray) -> np.ndarray:
    quat = np.zeros((len(yaw), 4), dtype=np.float64)
    quat[:, 2] = np.sin(0.5 * yaw)
    quat[:, 3] = np.cos(0.5 * yaw)
    return quat


def _max_true_run_seconds(mask: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros(mask.shape[1], dtype=np.float64)
    for joint in range(mask.shape[1]):
        best = current = 0
        for active in mask[:, joint]:
            current = current + 1 if active else 0
            best = max(best, current)
        result[joint] = best / fps
    return result


def _dynamic_metrics(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    velocity_limits: np.ndarray,
    fps: float,
) -> dict[str, np.ndarray | float]:
    joint_vel = _gradient(joint_pos, fps)
    speed_ratio = np.abs(joint_vel) / velocity_limits[None, :]
    root_xy_speed = np.linalg.norm(_gradient(root_pos[:, :2], fps), axis=1)
    yaw_rate = np.abs(_gradient(_yaw_from_quat_xyzw(root_quat), fps))
    return {
        "speed_ratio": speed_ratio,
        "speed_ratio_p99": np.quantile(speed_ratio, 0.99, axis=0),
        "speed_ratio_max": np.max(speed_ratio, axis=0),
        "speed_violation_fraction": float(np.mean(speed_ratio > 1.0)),
        "root_xy_speed_p99": float(np.quantile(root_xy_speed, 0.99)),
        "root_xy_speed_max": float(np.max(root_xy_speed)),
        "root_yaw_rate_p99": float(np.quantile(yaw_rate, 0.99)),
        "root_yaw_rate_max": float(np.max(yaw_rate)),
    }


def _resample_time(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    stretch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stretch <= 1.0 + 1.0e-6:
        return joint_pos, root_pos, root_quat
    output_frames = int(math.ceil((len(joint_pos) - 1) * stretch)) + 1
    source_u = np.linspace(0.0, 1.0, len(joint_pos))
    output_u = np.linspace(0.0, 1.0, output_frames)
    joint_out = np.column_stack(
        [np.interp(output_u, source_u, joint_pos[:, index]) for index in range(joint_pos.shape[1])]
    )
    root_out = np.column_stack(
        [np.interp(output_u, source_u, root_pos[:, index]) for index in range(3)]
    )
    yaw_out = np.interp(output_u, source_u, _yaw_from_quat_xyzw(root_quat))
    return joint_out, root_out, _quat_xyzw_from_yaw(yaw_out)


def audit_and_time_scale(
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    velocity_limits: np.ndarray,
    fps: float,
    auto_time_scale: bool,
    max_time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    initial = _dynamic_metrics(joint_pos, root_pos, root_quat, velocity_limits, fps)
    requested = max(
        1.0,
        float(np.max(initial["speed_ratio_p99"])) / 0.9,
        float(np.max(initial["speed_ratio_max"])) / 1.2,
        float(initial["root_xy_speed_p99"]) / 0.6,
        float(initial["root_xy_speed_max"]) / 0.8,
        float(initial["root_yaw_rate_p99"]) / 2.0,
        float(initial["root_yaw_rate_max"]) / 3.0,
    )
    applied = min(requested, max_time_scale) if auto_time_scale else 1.0
    joint_pos, root_pos, root_quat = _resample_time(joint_pos, root_pos, root_quat, applied)

    margin = np.maximum(1.0e-3, 0.01 * (upper - lower))
    near_limit = ((joint_pos - lower[None, :]) <= margin[None, :]) | (
        (upper[None, :] - joint_pos) <= margin[None, :]
    )
    near_fraction = near_limit.mean(axis=0)
    near_run_s = _max_true_run_seconds(near_limit, fps)
    final = _dynamic_metrics(joint_pos, root_pos, root_quat, velocity_limits, fps)

    reasons = []
    if float(np.max(near_fraction)) > 0.05:
        reasons.append("joint_near_limit_fraction")
    if float(np.max(near_run_s)) > 0.25:
        reasons.append("joint_near_limit_run")
    if float(final["speed_violation_fraction"]) > 0.005:
        reasons.append("joint_speed_violation_fraction")
    if float(np.max(final["speed_ratio_p99"])) > 1.0:
        reasons.append("joint_speed_p99")
    if float(np.max(final["speed_ratio_max"])) > 1.2:
        reasons.append("joint_speed_max")
    if float(final["root_xy_speed_p99"]) > 0.6 or float(final["root_xy_speed_max"]) > 0.8:
        reasons.append("root_xy_speed")
    if float(final["root_yaw_rate_p99"]) > 2.0 or float(final["root_yaw_rate_max"]) > 3.0:
        reasons.append("root_yaw_rate")
    if requested > max_time_scale + 1.0e-6:
        reasons.append("required_time_scale_exceeds_cap")

    quality = {
        "quality_pass": not reasons,
        "reasons": reasons,
        "time_scale_requested": requested,
        "time_scale_applied": applied,
        "near_limit_fraction_by_joint": dict(zip(CUSTOM_JOINT_NAMES, near_fraction.tolist())),
        "near_limit_max_run_s_by_joint": dict(zip(CUSTOM_JOINT_NAMES, near_run_s.tolist())),
        "velocity_ratio_p99_by_joint": dict(zip(CUSTOM_JOINT_NAMES, final["speed_ratio_p99"].tolist())),
        "velocity_ratio_max_by_joint": dict(zip(CUSTOM_JOINT_NAMES, final["speed_ratio_max"].tolist())),
        "speed_violation_fraction": float(final["speed_violation_fraction"]),
        "root_xy_speed_p99": float(final["root_xy_speed_p99"]),
        "root_xy_speed_max": float(final["root_xy_speed_max"]),
        "root_yaw_rate_p99": float(final["root_yaw_rate_p99"]),
        "root_yaw_rate_max": float(final["root_yaw_rate_max"]),
    }
    return joint_pos, root_pos, root_quat, quality


def iter_inputs(path: Path, pattern: str, limit: int | None) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No HumanML3D clips found at {path} with pattern {pattern}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ik-root", type=Path, default=DEFAULT_IK_ROOT)
    parser.add_argument("--training-urdf", type=Path, default=DEFAULT_TRAINING_URDF)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--root-scale", type=float, default=0.28963)
    parser.add_argument("--nominal-root-height", type=float, default=0.28)
    parser.add_argument("--max-time-scale", type=float, default=6.0)
    parser.add_argument("--auto-time-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-quality-failures", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")
    if args.max_time_scale < 1.0:
        raise ValueError("--max-time-scale must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lower, upper, velocity_limits = load_limits(args.training_urdf)
    ik = load_ik_module(args.ik_root)
    old_names = list(ik.H1Config.H1_JOINT_NAMES)
    inputs = iter_inputs(args.input, args.pattern, args.limit)

    motions, failures = [], []
    for index, path in enumerate(inputs, start=1):
        motion_id = path.stem
        print(f"[{index}/{len(inputs)}] retarget {motion_id}", flush=True)
        try:
            joints = validate_humanml3d(np.load(path, allow_pickle=False), path)
            solver = ik.H1PinkSolver(str(args.ik_root / "data/urdf/Assembly.urdf"))
            solver.TARGET_FPS = args.output_fps
            old_result = solver.process_motion(joints, head_rot_mats=None, visualize=False)
            old_dof = np.asarray(old_result["dof"], dtype=np.float64)
            joint_pos = map_28_to_30(old_dof, old_names)
            unclipped = joint_pos.copy()
            joint_pos = np.clip(joint_pos, lower, upper)
            clipped_fraction = float(np.mean(np.abs(unclipped - joint_pos) > 1.0e-7))
            root_pos, root_quat = restore_root_trajectory(
                joints, joint_pos.shape[0], args.root_scale, args.nominal_root_height
            )
            joint_pos, root_pos, root_quat, quality = audit_and_time_scale(
                joint_pos,
                root_pos,
                root_quat,
                lower,
                upper,
                velocity_limits,
                args.output_fps,
                args.auto_time_scale,
                args.max_time_scale,
            )
            if not quality["quality_pass"] and not args.allow_quality_failures:
                raise ValueError(f"quality gate failed: {quality['reasons']}")

            output = args.output_dir / f"{motion_id}.retarget.npz"
            np.savez_compressed(
                output,
                schema_version=np.asarray([1], dtype=np.int32),
                fps=np.asarray([args.output_fps], dtype=np.float32),
                source_fps=np.asarray([20], dtype=np.float32),
                source_id=np.asarray([motion_id]),
                joint_names=np.asarray(CUSTOM_JOINT_NAMES),
                joint_pos=joint_pos.astype(np.float32),
                root_pos=root_pos.astype(np.float32),
                root_quat_xyzw=root_quat.astype(np.float32),
                mapping_method=np.asarray(["ik_v2_28dof_to_custom30_bootstrap"]),
                clipped_fraction=np.asarray([clipped_fraction], dtype=np.float32),
                quality_pass=np.asarray([quality["quality_pass"]], dtype=np.bool_),
                quality_json=np.asarray([json.dumps(quality, sort_keys=True)]),
            )
            motions.append({
                "id": motion_id,
                "file": str(output.resolve()),
                "weight": 1.0,
                "source_file": str(path.resolve()),
                "frames": int(joint_pos.shape[0]),
                "clipped_fraction": clipped_fraction,
                "quality_pass": bool(quality["quality_pass"]),
                "quality": quality,
            })
        except Exception as exc:
            failures.append({"id": motion_id, "file": str(path), "error": str(exc)})
            traceback.print_exc()
            if not args.continue_on_error:
                raise

    manifest = {
        "schema_version": 1,
        "stage": "retargeted_joint_motion",
        "target_fps": args.output_fps,
        "joint_names": CUSTOM_JOINT_NAMES,
        "quality_gate_version": 1,
        "motions": motions,
        "failures": failures,
    }
    manifest_path = args.output_dir.parent / "retarget_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(motions)} motions and {len(failures)} failures to {manifest_path}")


if __name__ == "__main__":
    main()
