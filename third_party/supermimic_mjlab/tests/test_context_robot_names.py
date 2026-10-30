"""Robot-specific context names must not silently create empty foot tensors."""

from types import SimpleNamespace
import pytest
import torch
from supermimic.algo import state


def fake_env(names):
    model = SimpleNamespace(
        ngeom=len(names), geom=lambda i: SimpleNamespace(name=names[i])
    )
    sim = SimpleNamespace(
        mj_model=model, get_default_field=lambda name: torch.zeros(1, 3)
    )
    return SimpleNamespace(scene={"robot": object()}, sim=sim, device="cpu")


def test_custom_robot_context_names(monkeypatch):
    seen = []
    monkeypatch.setattr(
        state, "_model_name_id", lambda model, kind, name: seen.append(name) or 0
    )
    env = fake_env(["robot/left_foot_collision", "robot/right_foot_collision"])
    spec = SimpleNamespace(device=torch.device("cpu"))
    x = state.StateExtractor(
        env,
        spec,
        torso_body_name="chest",
        left_foot_geom_pattern="^left_foot_collision$",
        right_foot_geom_pattern="^right_foot_collision$",
    )
    assert seen == ["chest"]
    assert x.left_foot_geom_ids.tolist() == [0]
    assert x.right_foot_geom_ids.tolist() == [1]


def test_missing_foot_context_rejected(monkeypatch):
    monkeypatch.setattr(state, "_model_name_id", lambda *args: 0)
    with pytest.raises(ValueError, match="must each match"):
        state.StateExtractor(
            fake_env(["some_unrelated_collision"]),
            SimpleNamespace(device=torch.device("cpu")),
        )
