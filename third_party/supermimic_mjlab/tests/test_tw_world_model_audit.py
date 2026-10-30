"""Check physical-unit rollout measurements against a known exact transition model."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from supermimic.config.tw.contract import CUSTOM_HUMANOID_30DOF_JOINT_NAMES

path = (
    Path(__file__).resolve().parents[3] / "scripts/practice9/evaluate_supermimic_tw.py"
)
spec = importlib.util.spec_from_file_location("tw_eval", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactDynamics:
    def __call__(self, state, action, context):
        result = state.clone()
        result[:, 0] += action[:, 0] * 0.02
        result[:, 13:43] += action * 0.02
        result[:, 43:73] = action
        return result


def fixture(count):
    model = ExactDynamics()
    trainer = SimpleNamespace(
        device="cpu",
        world_model=model,
        env=SimpleNamespace(step_dt=0.02),
        spec=SimpleNamespace(joint_names=CUSTOM_HUMANOID_30DOF_JOINT_NAMES),
    )
    state = torch.zeros(1, 73)
    state[:, 3] = 1
    rows = {k: [] for k in ("state", "next_state", "action", "context")}
    for step in range(count):
        action = torch.full((1, 30), 0.01 * (step + 1))
        context = torch.zeros(1, 34)
        next_state = model(state, action, context)
        for key, value in [
            ("state", state),
            ("next_state", next_state),
            ("action", action),
            ("context", context),
        ]:
            rows[key].append(value[0].numpy().copy())
        state = next_state
    return trainer, rows


def test_open_loop_horizons_match_recorded_dynamics_without_off_by_one():
    trainer, rows = fixture(36)
    result = module.audit_world_model(trainer, rows)
    for horizon, windows in [(1, 36), (8, 29), (32, 5)]:
        row = result[str(horizon)]
        assert row["windows"] == windows
        assert row["finite_fraction"] == 1.0
        assert row["root_position_error_mean_m"] < 1e-6
        assert row["joint_position_rmse_rad"] < 1e-6
        assert row["joint_velocity_rmse_rad_s"] < 1e-6
        assert row["root_orientation_error_p95_rad"] < 1e-6


def test_measurement_keeps_meters_radians_and_short_episodes():
    trainer, rows = fixture(3)
    for state in rows["next_state"]:
        state[0] += 1.0
        state[13:43] += 0.5
        state[3:7] = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
        state[7:10] += 0.25
        state[10:13] += 0.75
    result = module.audit_world_model(trainer, rows)
    np.testing.assert_allclose(
        result["1"]["root_position_error_mean_m"], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(result["1"]["joint_position_rmse_rad"], 0.5, atol=1e-6)
    np.testing.assert_allclose(
        result["1"]["root_orientation_error_p95_rad"], np.pi / 2, atol=1e-6
    )
    np.testing.assert_allclose(
        result["1"]["root_linear_velocity_rmse_m_s"], 0.25, atol=1e-6
    )
    np.testing.assert_allclose(
        result["1"]["root_angular_velocity_rmse_rad_s"], 0.75, atol=1e-6
    )
    assert result["8"]["windows"] == 0 and result["32"]["windows"] == 0


def test_nonfinite_imagination_is_reported_as_failure():
    trainer, rows = fixture(2)
    trainer.world_model = lambda state, action, context: state * float("nan")
    result = module.audit_world_model(trainer, rows)["1"]
    assert result["finite_fraction"] == 0.0
    assert "failure" in result
    assert "root_position_error_mean_m" not in result
