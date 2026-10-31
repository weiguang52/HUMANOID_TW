"""Generate a synthetic default-pose calibration reference; never training/test data."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from supermimic.config.tw.contract import (
    CUSTOM_HUMANOID_TRACKING_BODY_NAMES,
    COORDINATES_VERSION,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm-roll-deg", type=float, default=90.0)
    p.add_argument("--pose-json", type=Path)
    a = p.parse_args()
    assert 0 <= a.arm_roll_deg <= 90
    assert not a.output.exists(), "Refusing to overwrite diagnostic reference"
    model = mujoco.MjModel.from_xml_path(str(a.asset))
    data = mujoco.MjData(model)
    assert model.nq == 37 and model.nv == 36
    data.qpos[:7] = [0.0, 0.0, 0.264, 1.0, 0.0, 0.0, 0.0]
    joints = [
        model.joint(j).name
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    default = {name: 0.0 for name in joints}
    default["left_shoulder_roll_joint"] = -math.radians(a.arm_roll_deg)
    default["right_shoulder_roll_joint"] = math.radians(a.arm_roll_deg)
    if a.pose_json:
        pose=json.loads(a.pose_json.read_text())
        assert pose['geometry_pass']
        assert set(pose['joint_targets']) <= set(joints)
        default.update(pose['joint_targets'])
        data.qpos[2]=pose['root_z_m']
        data.qpos[0]=pose.get('root_x_m',0.0)
    for name in joints:
        data.qpos[model.joint(name).qposadr[0]] = default[name]
    mujoco.mj_forward(model, data)
    bodies = list(CUSTOM_HUMANOID_TRACKING_BODY_NAMES)
    body_ids = [model.body(name).id for name in bodies]
    count = 1001
    repeat = lambda x: np.repeat(np.asarray(x, dtype=np.float32)[None], count, axis=0)
    a.output.mkdir(parents=True)
    path = a.output / "synthetic_static_default.npz"
    np.savez_compressed(
        path,
        fps=np.array([50.0], dtype=np.float32),
        joint_names=np.asarray(joints),
        body_names=np.asarray(bodies),
        joint_coordinate_contract=np.asarray([COORDINATES_VERSION]),
        joint_pos=repeat([default[name] for name in joints]),
        joint_vel=np.zeros((count, 30), dtype=np.float32),
        body_pos_w=repeat(data.xpos[body_ids]),
        body_quat_w=repeat(data.xquat[body_ids]),
        body_lin_vel_w=np.zeros((count, len(bodies), 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((count, len(bodies), 3), dtype=np.float32),
        root_pos=repeat(data.xpos[model.body("base_link").id]),
        root_quat_wxyz=repeat(data.xquat[model.body("base_link").id]),
    )
    manifest = dict(
        synthetic_diagnostic=True,
        arm_roll_abs_deg=a.arm_roll_deg,
        pose_override_sha256=hashlib.sha256(a.pose_json.read_bytes()).hexdigest() if a.pose_json else None,
        joint_coordinate_contract=COORDINATES_VERSION,
        target_fps=50,
        source_asset=str(a.asset),
        source_asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),
        note="Synthetic explicit pose. No HumanML3D source. Separate from the fixed332 training and7 test motions; cannot count as learned-policy success.",
        motions=[
            dict(
                id=path.stem,
                file=str(path),
                frames=count,
                split="diagnostic",
                category="synthetic_static_calibration",
            )
        ],
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
