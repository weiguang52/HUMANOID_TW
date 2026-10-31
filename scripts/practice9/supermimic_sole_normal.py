"""Reference-normal reward term for the explicitly gated static PPO diagnostic."""

import math

import torch
import torch.nn.functional as F


def normals_from_quat(quat):
    q = F.normalize(quat, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )


def normal_alignment_cost(actual, reference):
    """Mean over the two feet of signed unit-normal squared distance, in [0,4]."""
    return (
        (normals_from_quat(actual) - normals_from_quat(reference))
        .square()
        .sum(-1)
        .mean(-1)
    )


def check_known_rotations():
    v = math.sqrt(0.5)
    q = torch.tensor(
        [[1, 0, 0, 0], [v, 0, 0, v], [v, v, 0, 0], [0, 1, 0, 0]], dtype=torch.float64
    )
    q = q[:, None].repeat(1, 2, 1).requires_grad_()
    ref = torch.zeros_like(q)
    ref[..., 0] = 1
    cost = normal_alignment_cost(q, ref)
    torch.testing.assert_close(
        cost,
        torch.tensor([0.0, 0.0, 2.0, 4.0], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    cost.sum().backward()
    assert torch.isfinite(q.grad).all()
    return {
        "identity_and_yaw_zero": True,
        "tilt90_cost2": True,
        "flipped180_cost4": True,
        "gradient_finite": True,
    }


class SoleNormalPenalty:
    def __init__(self, trainer, coefficient):
        import mujoco
        import numpy as np

        assert math.isfinite(coefficient) and coefficient > 0
        self.t = trainer
        self.coefficient = coefficient
        self.max_native_error = 0.0
        self.calls = 0
        names = trainer.command.cfg.body_names
        self.foot_ids = [list(names).index(n) for n in ["left_foot", "right_foot"]]
        model = trainer.env.sim.mj_model
        self.geom_ids = []
        for side in ["left", "right"]:
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{side}_foot_collision"
            )
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{side}_foot"
            )
            assert gid >= 0 and bid >= 0 and model.geom_bodyid[gid] == bid
            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
            np.testing.assert_array_equal(model.geom_quat[gid], [1, 0, 0, 0])
            self.geom_ids.append(gid)

    @torch.no_grad()
    def __call__(self, reference):
        t = self.t
        q = t.command.robot_body_quat_w[:, self.foot_ids]
        ref = reference["body_quat"][:, self.foot_ids]
        predicted = normals_from_quat(q)
        native = t.env.sim.data.geom_xmat[:, self.geom_ids].reshape(
            t.env.num_envs, 2, 3, 3
        )[..., :, 2]
        error = float((native - predicted).abs().max())
        self.max_native_error = max(self.max_native_error, error)
        torch.testing.assert_close(predicted, native, rtol=0, atol=1e-6)
        cost = normal_alignment_cost(q, ref)
        assert torch.isfinite(cost).all() and bool(
            ((cost >= 0) & (cost <= 4 + 1e-6)).all()
        )
        self.calls += 1
        return self.coefficient * t.env.step_dt * cost, cost
