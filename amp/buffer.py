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
    ) -> None:
        self.num_envs  = num_envs
        self.num_steps = num_steps
        self.device    = device

        self.obs       = torch.zeros(num_steps, num_envs, obs_dim,        device=device)
        self.actions   = torch.zeros(num_steps, num_envs, action_dim,     device=device)
        self.log_probs = torch.zeros(num_steps, num_envs,                 device=device)
        self.rewards   = torch.zeros(num_steps, num_envs,                 device=device)
        self.values    = torch.zeros(num_steps, num_envs,                 device=device)
        self.dones     = torch.zeros(num_steps, num_envs,                 device=device)
        self.amp_obs   = torch.zeros(num_steps, num_envs, amp_obs_dim,    device=device)

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
        dones: Tensor,
        amp_obs: Tensor,
    ) -> None:
        self.obs[step]       = obs
        self.actions[step]   = actions
        self.log_probs[step] = log_probs
        self.rewards[step]   = rewards
        self.values[step]    = values
        self.dones[step]     = dones.float()
        self.amp_obs[step]   = amp_obs

    def compute_returns(
        self,
        last_values: Tensor,
        gamma: float = 0.99,
        lam: float   = 0.95,
    ) -> None:
        """GAE-Lambda advantage estimation.

        delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t       = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
        returns_t = A_t + V(s_t)
        """
        adv   = torch.zeros_like(self.rewards)
        gae   = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_val   = last_values
                next_nterm = 1.0 - self.dones[t]
            else:
                next_val   = self.values[t + 1]
                next_nterm = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_val * next_nterm - self.values[t]
            gae   = delta + gamma * lam * next_nterm * gae
            adv[t] = gae

        self.advantages = adv
        self.returns    = adv + self.values

    def get_minibatches(
        self, num_mini_batches: int
    ) -> Generator[dict[str, Tensor], None, None]:
        """Flatten, shuffle, and yield minibatches.

        Yields dicts with keys:
            obs, actions, log_probs_old, returns, advantages, amp_obs
        """
        assert self.advantages is not None, "Call compute_returns() first"

        total = self.num_steps * self.num_envs
        idx   = torch.randperm(total, device=self.device)
        batch_size = total // num_mini_batches

        # Flatten time × env → total
        flat = {
            "obs":           self.obs.view(total, -1),
            "actions":       self.actions.view(total, -1),
            "log_probs_old": self.log_probs.view(total),
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
        B = amp_obs.shape[0]
        end = self.ptr + B
        if end <= self.capacity:
            self.data[self.ptr : end] = amp_obs
        else:
            # Wrap around
            first = self.capacity - self.ptr
            self.data[self.ptr :] = amp_obs[:first]
            self.data[: B - first] = amp_obs[first:]
        self.ptr  = end % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int) -> Tensor:
        """Sample a random batch.  Returns (batch_size, amp_obs_dim)."""
        assert self.size > 0, "Buffer is empty"
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.data[idx]

    def __len__(self) -> int:
        return self.size
