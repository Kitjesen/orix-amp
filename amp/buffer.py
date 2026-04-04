# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Rollout buffer for PPO and replay buffer for AMP discriminator.

No external RL library dependencies — pure PyTorch.
"""
from __future__ import annotations

from typing import Generator
import torch
from torch import Tensor


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Fixed-length on-policy rollout storage for PPO.

    Stores num_steps transitions across num_envs parallel environments,
    then computes GAE-Lambda advantages in place.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        amp_obs_dim: int,
        device: torch.device,
        critic_obs_dim: int | None = None,
    ) -> None:
        self.num_envs  = num_envs
        self.num_steps = num_steps
        self.device    = device

        c_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim
        self.obs          = torch.zeros(num_steps, num_envs, obs_dim,     device=device)
        self.critic_obs   = torch.zeros(num_steps, num_envs, c_dim,       device=device)
        self.actions      = torch.zeros(num_steps, num_envs, action_dim,  device=device)
        self.log_probs    = torch.zeros(num_steps, num_envs,              device=device)
        self.rewards      = torch.zeros(num_steps, num_envs,              device=device)
        self.values       = torch.zeros(num_steps, num_envs,              device=device)
        self.terminations = torch.zeros(num_steps, num_envs,              device=device)
        self.amp_obs      = torch.zeros(num_steps, num_envs, amp_obs_dim, device=device)

        # Computed by compute_returns()
        self.advantages: Tensor | None = None
        self.returns:    Tensor | None = None

    def add(
        self,
        step: int,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        values: Tensor,
        terminated: Tensor,
        amp_obs: Tensor,
        critic_obs: Tensor | None = None,
    ) -> None:
        self.obs[step]          = obs
        self.critic_obs[step]   = critic_obs if critic_obs is not None else obs
        self.actions[step]      = actions
        self.log_probs[step]    = log_probs
        self.rewards[step]      = rewards
        self.values[step]       = values
        self.terminations[step] = terminated.float()
        self.amp_obs[step]      = amp_obs

    def compute_returns(
        self,
        last_values: Tensor,
        gamma: float = 0.99,
        lam: float   = 0.95,
    ) -> None:
        """GAE-Lambda advantage estimation.

        Uses terminations (not time-limit truncations) to mask next-state value.
        This correctly handles episode timeout: we still bootstrap V(s_{T+1}).

        delta_t   = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
        A_t       = delta_t + gamma * lam * (1 - terminated_t) * A_{t+1}
        returns_t = A_t + V(s_t)
        """
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            not_term = 1.0 - self.terminations[t]
            next_val = last_values if t == self.num_steps - 1 else self.values[t + 1]

            delta  = self.rewards[t] + gamma * next_val * not_term - self.values[t]
            gae    = delta + gamma * lam * not_term * gae
            adv[t] = gae

        self.advantages = adv
        self.returns    = adv + self.values

    def get_minibatches(
        self, num_mini_batches: int
    ) -> Generator[dict[str, Tensor], None, None]:
        """Flatten, shuffle, and yield minibatches.

        Yields dicts with keys:
            obs, actions, log_probs_old, values_old, returns, advantages, amp_obs
        """
        assert self.advantages is not None, "Call compute_returns() first"

        total      = self.num_steps * self.num_envs
        idx        = torch.randperm(total, device=self.device)
        batch_size = total // num_mini_batches

        flat = {
            "obs":           self.obs.view(total, -1),
            "critic_obs":    self.critic_obs.view(total, -1),
            "actions":       self.actions.view(total, -1),
            "log_probs_old": self.log_probs.view(total),
            "values_old":    self.values.view(total),
            "returns":       self.returns.view(total),          # type: ignore[union-attr]
            "advantages":    self.advantages.view(total),
            "amp_obs":       self.amp_obs.view(total, -1),
        }

        for start in range(0, total, batch_size):
            mb_idx = idx[start : start + batch_size]
            yield {k: v[mb_idx] for k, v in flat.items()}


# ── AMP Replay Buffer ─────────────────────────────────────────────────────────

class AMPReplayBuffer:
    """Circular replay buffer for AMP observation tensors.

    Used for both expert reference motion samples and recent agent transitions.
    """

    def __init__(
        self,
        capacity: int,
        amp_obs_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity    = capacity
        self.amp_obs_dim = amp_obs_dim
        self.device      = device

        self.data = torch.zeros(capacity, amp_obs_dim, device=device)
        self.ptr  = 0
        self.size = 0

    def add(self, amp_obs: Tensor) -> None:
        """Add a batch of AMP observations.  amp_obs: (B, amp_obs_dim)"""
        B   = amp_obs.shape[0]
        end = self.ptr + B
        if end <= self.capacity:
            self.data[self.ptr : end] = amp_obs
        else:
            first = self.capacity - self.ptr
            self.data[self.ptr :]    = amp_obs[:first]
            self.data[: B - first]   = amp_obs[first:]
        self.ptr  = end % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int) -> Tensor:
        """Sample a random batch.  Returns (batch_size, amp_obs_dim)."""
        assert self.size > 0, "Buffer is empty"
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.data[idx]

    def __len__(self) -> int:
        return self.size
