"""Rebuild TW reference FK and velocities from prescribed poses, without export.

Positions use body origins; linear velocities use body centers of mass, matching
existing TW references. Angular velocities are world-frame central differences.
This helper neither optimizes poses nor approves physical feasibility.
"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def reconstruct(source, model, joint_positions, root_positions):
    joints = np.asarray(joint_positions, dtype=np.float64)
    roots = np.asarray(root_positions, dtype=np.float64)
    n = len(source['joint_pos'])
    assert n >= 3 and joints.shape == (n, 30) and roots.shape == (n, 3)
    assert np.isfinite(joints).all() and np.isfinite(roots).all()
    dt = 1 / float(source['fps'].item())
    assert dt > 0
    names = source['joint_names'].tolist()
    qa = [int(model.joint(str(name)).qposadr[0]) for name in names]
    bodies = [model.body(str(name)).id for name in source['body_names']]
    assert len(set(qa)) == 30 and len(set(bodies)) == len(bodies)
    data = mujoco.MjData(model)
    positions, centers, quaternions = [], [], []
    for i in range(n):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = roots[i]
        data.qpos[3:7] = source['root_quat_wxyz'][i]
        data.qpos[qa] = joints[i]
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[bodies].copy())
        centers.append(data.xipos[bodies].copy())
        quaternions.append(data.xquat[bodies].copy())
    positions, centers, quaternions = map(np.asarray, (positions, centers, quaternions))
    for i in range(1, n):
        flip = np.sum(quaternions[i] * quaternions[i-1], axis=-1) < 0
        quaternions[i, flip] *= -1
    angular = np.zeros_like(positions)
    for body in range(len(bodies)):
        rotations = Rotation.from_quat(quaternions[:, body][:, [1, 2, 3, 0]])
        central = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2*dt)
        angular[:, body] = np.vstack((central[:1], central, central[-1:]))
    updates = dict(joint_pos=joints, root_pos=roots,
                   joint_vel=np.gradient(joints, dt, axis=0),
                   body_pos_w=positions, body_quat_w=quaternions,
                   body_lin_vel_w=np.gradient(centers, dt, axis=0),
                   body_ang_vel_w=angular)
    output = {key: value.copy() for key, value in source.items()}
    for key, value in updates.items():
        assert value.shape == source[key].shape and np.isfinite(value).all()
        output[key] = value.astype(source[key].dtype)
    assert set(output) == set(source)
    for key in set(source) - set(updates):
        assert np.array_equal(source[key], output[key])
    return output
