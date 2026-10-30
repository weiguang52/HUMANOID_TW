import numpy as np
import pytest
from supermimic.mdp.motion_library import MotionLibrary, MotionLibraryCfg


def write_motion(p, contract="tw123_yaw_waist_v3", names=("b", "a")):
    np.savez(
        p,
        fps=np.array([50.0]),
        joint_names=np.array(names),
        body_names=np.array(["base_link"]),
        joint_coordinate_contract=np.array([contract]),
        joint_pos=np.array([[10.0, 20.0], [11.0, 21.0]]),
        joint_vel=np.array([[30.0, 40.0], [31.0, 41.0]]),
        body_pos_w=np.zeros((2, 1, 3)),
        body_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]] * 2),
        body_lin_vel_w=np.zeros((2, 1, 3)),
        body_ang_vel_w=np.zeros((2, 1, 3)),
    )


def cfg(p):
    return MotionLibraryCfg(
        motion_files=(str(p),),
        body_names=("base_link",),
        device="cpu",
        joint_names=("a", "b"),
        required_coordinate_contract="tw123_yaw_waist_v3",
    )


def test_joint_columns_follow_names(tmp_path):
    p = tmp_path / "motion.npz"
    write_motion(p)
    lib = MotionLibrary(cfg(p))
    np.testing.assert_array_equal(lib.joint_pos.numpy(), [[20.0, 10.0], [21.0, 11.0]])
    np.testing.assert_array_equal(lib.joint_vel.numpy(), [[40.0, 30.0], [41.0, 31.0]])


def test_stale_coordinates_rejected(tmp_path):
    p = tmp_path / "motion.npz"
    write_motion(p, contract="old")
    with pytest.raises(ValueError, match="stale"):
        MotionLibrary(cfg(p))


def test_duplicate_joint_metadata_rejected(tmp_path):
    p = tmp_path / "motion.npz"
    write_motion(p, names=("a", "a"))
    with pytest.raises(ValueError, match="duplicates"):
        MotionLibrary(cfg(p))


def test_nonfinite_motion_rejected(tmp_path):
    p = tmp_path / "motion.npz"
    write_motion(p)
    with np.load(p, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    values["joint_vel"][0, 0] = np.nan
    np.savez(p, **values)
    with pytest.raises(ValueError, match="nonfinite"):
        MotionLibrary(cfg(p))


def test_nonunit_quaternion_rejected(tmp_path):
    p = tmp_path / "motion.npz"
    write_motion(p)
    with np.load(p, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    values["body_quat_w"] *= 2
    np.savez(p, **values)
    with pytest.raises(ValueError, match="quaternions"):
        MotionLibrary(cfg(p))
