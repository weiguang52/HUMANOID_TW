"""Per-environment ring buffer of simulator transitions."""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        capacity: int,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        device: str | torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.device(device)
        self.states = torch.zeros(num_envs, capacity, state_dim, device=self.device)
        self.actions = torch.zeros(num_envs, capacity, action_dim, device=self.device)
        self.contexts = torch.zeros(num_envs, capacity, context_dim, device=self.device)
        self.rewards = torch.zeros(num_envs, capacity, device=self.device)
        self.dones = torch.zeros(
            num_envs, capacity, dtype=torch.bool, device=self.device
        )
        self.terminations = torch.zeros(
            num_envs, capacity, dtype=torch.bool, device=self.device
        )
        self.clip_ids = torch.zeros(
            num_envs, capacity, dtype=torch.int32, device=self.device
        )
        self.time_steps = torch.zeros(
            num_envs, capacity, dtype=torch.long, device=self.device
        )
        self._window_cache = {}
        self.valid_window_counts = {}
        self.last_cross_reset_count = 0
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def numel(self) -> int:
        return int(self.count.sum().item())

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        context: torch.Tensor,
        clip_id: torch.Tensor,
        time_step: torch.Tensor,
        terminated: torch.Tensor | None = None,
    ) -> None:
        self._window_cache.clear()
        self.valid_window_counts.clear()
        idx = self.ptr
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.states[env_ids, idx] = state
        self.actions[env_ids, idx] = action
        self.contexts[env_ids, idx] = context
        self.rewards[env_ids, idx] = reward
        self.dones[env_ids, idx] = done
        # Hard simulator termination (height/tilt), as opposed to a time-out.
        self.terminations[env_ids, idx] = done if terminated is None else terminated
        self.clip_ids[env_ids, idx] = clip_id.to(torch.int32)
        self.time_steps[env_ids, idx] = time_step
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = torch.minimum(
            self.count + 1, torch.full_like(self.count, self.capacity)
        )

    def _valid_len(self) -> torch.Tensor:
        return self.count

    def sample_windows(
        self, batch_size: int, length: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(env_id, end_position)`` pairs for valid windows.

        The returned position is the last state of the window, so
        ``gather_states(env_ids, positions, before=length - 1)`` yields the
        window in chronological order.
        """
        if not 1 <= length <= self.capacity:
            raise ValueError("Replay window length must fit capacity")
        if length not in self._window_cache:
            offsets = torch.arange(self.capacity, device=self.device)
            oldest = self.ptr - self.count
            chronological = (oldest[:, None] + offsets[None, :]) % self.capacity
            ordered_dones = self.dones.gather(1, chronological)
            prefix = torch.cat(
                [
                    torch.zeros(
                        (self.num_envs, 1), dtype=torch.long, device=self.device
                    ),
                    ordered_dones.long().cumsum(dim=1),
                ],
                dim=1,
            )
            boundary_end = (offsets + length - 1).clamp_max(self.capacity)
            crossed = prefix[:, boundary_end] - prefix[:, offsets]
            live = offsets[None, :] + length <= self.count[:, None]
            valid = ((crossed == 0) & live).nonzero(as_tuple=False)
            if valid.shape[0] == 0:
                raise RuntimeError(
                    f"No episode-contiguous replay windows of length {length}; collect more valid transitions"
                )
            envs, starts = valid[:, 0], valid[:, 1]
            ends = (oldest[envs] + starts + length - 1) % self.capacity
            self._window_cache[length] = (envs, ends)
            self.valid_window_counts[length] = valid.shape[0]
        envs, ends = self._window_cache[length]
        selected = torch.randint(
            0, envs.shape[0], (batch_size,), device=self.device, generator=generator
        )
        env_ids, positions = envs[selected], ends[selected]
        # Verify the actual returned batch, including ring-wrap cases.
        prior = (
            positions[:, None]
            + torch.arange(1 - length, 0, device=self.device)[None, :]
        ) % self.capacity
        self.last_cross_reset_count = int(
            self.dones[env_ids[:, None], prior].any(dim=1).sum().item()
        )
        if self.last_cross_reset_count:
            raise RuntimeError(
                "Internal replay error: sampled windows cross an episode reset"
            )
        return env_ids, positions

    def sample_starts(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        env_ids = torch.randint(
            0, self.num_envs, (batch_size,), device=self.device, generator=generator
        )
        valid_len = self._valid_len()[env_ids].clamp_min(1)
        offset = (torch.rand(batch_size, device=self.device) * valid_len.float()).long()
        pos = (self.ptr[env_ids] - valid_len + offset) % self.capacity
        return {
            "env_ids": env_ids,
            "positions": pos,
            "state": self.states[env_ids, pos],
            "context": self.contexts[env_ids, pos],
            "clip_ids": self.clip_ids[env_ids, pos].long(),
            "time_steps": self.time_steps[env_ids, pos],
        }

    def gather_states(
        self, env_ids: torch.Tensor, positions: torch.Tensor, before: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = torch.arange(-before, 1, device=self.device)
        idx = (positions[:, None] + offsets[None, :]) % self.capacity
        return (
            self.states[env_ids[:, None], idx],
            self.actions[env_ids[:, None], idx],
            self.dones[env_ids[:, None], idx],
        )

    def sample_random(self, n: int) -> dict[str, torch.Tensor]:
        flat = torch.randint(0, self.num_envs * self.capacity, (n,), device=self.device)
        env_ids = flat // self.capacity
        pos = flat % self.capacity
        live = pos < self.count[env_ids]
        return {
            "state": self.states[env_ids, pos],
            "action": self.actions[env_ids, pos],
            "context": self.contexts[env_ids, pos],
            "clip_ids": self.clip_ids[env_ids, pos].long(),
            "time_steps": self.time_steps[env_ids, pos],
            "terminated": self.terminations[env_ids, pos],
            "live": live,
        }
