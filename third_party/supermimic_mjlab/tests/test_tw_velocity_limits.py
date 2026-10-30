"""Physical speed caps: preserve floating-base acceleration and cap both signs."""

import numpy as np
import warp as wp
from supermimic.config.tw.velocity_limits import _bounded_acceleration, LIMITS


def test_velocity_projection_before_position_integration():
    velocity = np.array(
        [[10.0, 1.0, -1.0, 0.0], [-5.0, -1.0, 1.0, 0.0]], dtype=np.float32
    )
    acceleration = np.array(
        [[100.0, 10000.0, -10000.0, 10.0], [-100.0, -10000.0, 10000.0, -10.0]],
        dtype=np.float32,
    )
    dt = np.array([0.005], dtype=np.float32)
    caps = np.array([0.0, 2.0, 3.0, 2.0], dtype=np.float32)
    with wp.ScopedDevice("cpu"):
        v = wp.array(velocity, dtype=float)
        a = wp.array(acceleration, dtype=float)
        output = wp.empty_like(a)
        wp.launch(
            _bounded_acceleration,
            dim=velocity.shape,
            inputs=[wp.array(dt), v, a, wp.array(caps)],
            outputs=[output],
        )
        projected_a = output.numpy()
    expected = np.clip(velocity + dt[0] * acceleration, -caps, caps)
    expected[:, 0] = velocity[:, 0] + dt[0] * acceleration[:, 0]
    np.testing.assert_allclose(projected_a[:, 0], acceleration[:, 0], rtol=0, atol=0)
    bounded_v = velocity + dt[0] * projected_a
    np.testing.assert_allclose(bounded_v, expected, atol=1e-6)
    # The integrator subsequently uses this velocity for positions, not the uncapped one.
    np.testing.assert_allclose(dt[0] * bounded_v, dt[0] * expected, atol=1e-8)


def test_original_tw_joint_group_caps():
    assert len(LIMITS) == 30
    assert sum(cap == 3.0 for cap in LIMITS.values()) == 14
    assert sum(cap == 2.0 for cap in LIMITS.values()) == 16
