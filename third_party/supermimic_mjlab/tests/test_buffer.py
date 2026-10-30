"""Replay-window sampling must respect episode boundaries."""

import torch

from supermimic.algo.buffer import ReplayBuffer


def test_windows_do_not_cross_episode_boundaries():
    buffer = ReplayBuffer(2, 8, 3, 2, 1, "cpu")
    for t in range(6):
        buffer.add(
            state=torch.full((2, 3), float(t)),
            action=torch.full((2, 2), float(t)),
            reward=torch.zeros(2),
            done=torch.tensor([t == 2, False]),
            context=torch.zeros(2, 1),
            clip_id=torch.zeros(2, dtype=torch.long),
            time_step=torch.full((2,), t, dtype=torch.long),
        )
    env_ids, positions = buffer.sample_windows(512, length=3)
    states, actions, dones = buffer.gather_states(env_ids, positions, before=2)
    assert states.shape == (512, 3, 3)
    assert actions.shape == (512, 3, 2)
    assert dones.shape == (512, 3)
    assert not dones[:, :2].any()


def test_ring_buffer_wraps():
    buffer = ReplayBuffer(1, 4, 2, 1, 1, "cpu")
    for t in range(10):
        buffer.add(
            state=torch.full((1, 2), float(t)),
            action=torch.full((1, 1), float(t)),
            reward=torch.zeros(1),
            done=torch.zeros(1, dtype=torch.bool),
            context=torch.zeros(1, 1),
            clip_id=torch.zeros(1, dtype=torch.long),
            time_step=torch.full((1,), t, dtype=torch.long),
        )
    env_ids, positions = buffer.sample_windows(64, length=4)
    states, _, _ = buffer.gather_states(env_ids, positions, before=3)
    assert states.shape == (64, 4, 2)
    assert int(buffer.count.item()) == 4


def test_short_episode_env_never_falls_back_across_resets():
    buffer = ReplayBuffer(2, 16, 3, 2, 1, "cpu")
    for t in range(12):
        buffer.add(
            torch.full((2, 3), float(t)),
            torch.zeros(2, 2),
            torch.zeros(2),
            torch.tensor([t % 2 == 1, False]),
            torch.zeros(2, 1),
            torch.zeros(2, dtype=torch.long),
            torch.full((2,), t, dtype=torch.long),
        )
    env_ids, positions = buffer.sample_windows(512, 9)
    states, _, dones = buffer.gather_states(env_ids, positions, before=8)
    assert (env_ids == 1).all()  # Only this environment has any valid 9-state windows.
    assert not dones[:, :-1].any()
    assert states.shape == (512, 9, 3)
    assert buffer.valid_window_counts[9] == 4
    assert buffer.last_cross_reset_count == 0


def test_no_valid_full_horizon_raises_instead_of_inventing_transitions():
    import pytest

    buffer = ReplayBuffer(1, 16, 3, 2, 1, "cpu")
    for t in range(12):
        buffer.add(
            torch.ones(1, 3) * t,
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.tensor([True]),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
        )
    with pytest.raises(RuntimeError, match="No episode-contiguous"):
        buffer.sample_windows(32, 9)


def test_cached_window_pool_invalidated_after_ring_overwrite():
    buffer = ReplayBuffer(1, 4, 3, 2, 1, "cpu")
    for t in range(4):
        buffer.add(
            torch.ones(1, 3) * t,
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.tensor([False]),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
        )
    buffer.sample_windows(8, 4)
    for t in range(4, 8):
        buffer.add(
            torch.ones(1, 3) * t,
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.tensor([False]),
            torch.zeros(1, 1),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
        )
    e, p = buffer.sample_windows(8, 4)
    states, _, _ = buffer.gather_states(e, p, before=3)
    torch.testing.assert_close(states[0, :, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
