"""Group mean PD tracking error cost in radians; no assumed motor rating."""
import torch


def group_target_error(env, joint_names, scale=0.25):
    if scale <= 0:
        raise ValueError('scale must be positive')
    robot = env.scene['robot']
    ids, names = robot.find_joints(joint_names, preserve_order=True)
    if len(ids) != len(joint_names):
        raise ValueError('Incomplete compensation joint group')
    error = robot.data.joint_pos_target[:, ids] - robot.data.joint_pos[:, ids]
    return torch.mean(torch.square(error / scale), dim=-1)
