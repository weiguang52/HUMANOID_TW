from types import SimpleNamespace
import torch
from supermimic.algo.buffer import ReplayBuffer
from supermimic.algo.trainer import SuperMimicTrainer
from supermimic.mdp.features import ProprioFeatures, ProprioNoiseCfg


def make_trainer(count=12, reset_at=None, capacity=16):
    trainer = object.__new__(SuperMimicTrainer)
    trainer.history_len = 3
    trainer.spec = SimpleNamespace(num_joints=30)
    trainer.buffer = ReplayBuffer(1, capacity, 73, 30, 34, "cpu")
    trainer.proprio = ProprioFeatures(
        30,
        torch.zeros(30),
        ProprioNoiseCfg(joint_pos=0, joint_vel=0, base_ang_vel=0, projected_gravity=0),
    )
    bias = torch.arange(30, dtype=torch.float32)[None] / 100
    context = torch.cat(
        [torch.ones(1, 1), bias, torch.tensor([[10.0, 20.0, 30.0]])], dim=1
    )
    for t in range(count):
        state = torch.zeros(1, 73)
        state[:, 3] = 1
        state[:, 13:43] = float(t)
        trainer.buffer.add(
            state,
            torch.full((1, 30), 100.0 + t),
            torch.zeros(1),
            torch.tensor([t == reset_at]),
            context,
            torch.zeros(1, dtype=torch.long),
            torch.tensor([t]),
        )
    return trainer, bias


def history(trainer, physical):
    return trainer._history_from_buffer(torch.tensor([0]), torch.tensor([physical]))


def test_past_only_frames_and_previous_actions_with_correct_bias():
    trainer, bias = make_trainer()
    result = history(trainer, 11)
    for j, t in enumerate([8.0, 9.0, 10.0]):
        torch.testing.assert_close(result[0, j, :30], torch.full((30,), t) + bias[0])
        torch.testing.assert_close(result[0, j, -30:], torch.full((30,), 100.0 + t - 1))
    # The actor appends frame11 once, so its first actual input is [9,10,11].
    assert result[0, -1, 0].item() == 10.0


def test_latest_reset_pads_history_with_new_episode_and_zero_previous_action():
    trainer, bias = make_trainer(reset_at=9)
    result = history(trainer, 11)
    torch.testing.assert_close(
        result[0, :, :30], (torch.full((30,), 10.0) + bias[0]).expand(3, -1)
    )
    assert (result[0, :, -30:] == 0).all()
    # Frame10 itself is the first state after the reset.
    result = history(trainer, 10)
    torch.testing.assert_close(
        result[0, :, :30], (torch.full((30,), 10.0) + bias[0]).expand(3, -1)
    )
    assert (result[0, :, -30:] == 0).all()


def test_oldest_available_state_is_padded_without_reading_future_ring_data():
    trainer, bias = make_trainer(count=8, capacity=4)
    # Chronological available frames4..7: physical0 is oldest frame4.
    result = history(trainer, 0)
    torch.testing.assert_close(
        result[0, :, :30], (torch.full((30,), 4.0) + bias[0]).expand(3, -1)
    )
    assert (result[0, :, -30:] == 0).all()


def test_single_available_frame_never_uses_uninitialized_quaternion_or_actions():
    trainer, bias = make_trainer(count=1)
    result = history(trainer, 0)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0, :, :30], bias.expand(3, -1))
    assert (result[0, :, -30:] == 0).all()


def test_oldest_start_does_not_use_newest_ring_action_as_previous_action():
    trainer, _ = make_trainer(count=8, capacity=4)
    trainer._rng = torch.Generator()
    b = trainer.buffer
    b.sample_starts = lambda batch_size, generator: dict(
        env_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        state=b.states[0, 0][None],
        context=b.contexts[0, 0][None],
        clip_ids=torch.tensor([0]),
        time_steps=torch.tensor([4]),
    )
    starts = trainer._sample_imagination_starts(1)
    assert (starts["prev_action"] == 0).all()
