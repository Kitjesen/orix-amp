# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Actor-Critic and AMP Discriminator networks.

No external RL library dependencies — pure PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[name]


def _mlp(dims: list[int], activation: str, output_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if output_activation or i < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


# ── Actor-Critic ──────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with a shared Normal distribution policy.

    actor uses policy obs (realizable on real robot).
    critic uses privileged obs (sim-only: + base_lin_vel, contact, height_scan).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden: list[int] = (512, 256, 128),
        critic_hidden: list[int] = (512, 256, 128),
        init_noise_std: float = 1.0,
        activation: str = "elu",
        critic_obs_dim: int | None = None,   # if None, uses same obs_dim as actor
    ) -> None:
        super().__init__()
        c_dim = critic_obs_dim if critic_obs_dim is not None else obs_dim
        self.actor = _mlp([obs_dim, *actor_hidden, action_dim], activation)
        self.critic = _mlp([c_dim, *critic_hidden, 1], activation)
        # Learnable log std (scalar, shared across all action dims)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), fill_value=torch.log(torch.tensor(init_noise_std)))
        )
        self._init_weights()

    def _init_weights(self) -> None:
        actor_out = self.actor[-1]   # the final Linear of the actor MLP
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is actor_out else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    # ── Forward helpers ──

    def _distribution(self, obs: Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ── Public API ──

    @torch.no_grad()
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action and compute log-prob.

        Returns:
            actions:   (N, action_dim)
            log_probs: (N,)
        """
        dist = self._distribution(obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    @torch.no_grad()
    def get_value(self, obs: Tensor) -> Tensor:
        """Critic value estimate. Returns (N,)."""
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions under current policy (with grad).

        Returns:
            values:    (N,)
            log_probs: (N,)
            entropy:   (N,)
        """
        dist = self._distribution(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        return values, log_probs, entropy


# ── AMP Discriminator ─────────────────────────────────────────────────────────

class AMPDiscriminator(nn.Module):
    """Discriminator that classifies real (expert) vs fake (agent) AMP observations.

    Outputs raw logits — sigmoid is applied externally where needed.
    """

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: list[int] = (1024, 512),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = _mlp([amp_obs_dim, *hidden_dims, 1], activation)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, amp_obs: Tensor) -> Tensor:
        """Returns raw logits of shape (B, 1)."""
        return self.net(amp_obs)

    def compute_style_reward(self, amp_obs: Tensor, reward_scale: float = 2.0) -> Tensor:
        """Compute AMP style reward from agent observations.

        r_style = reward_scale * sigmoid(logit)  ∈ [0, reward_scale]
        High when discriminator thinks agent looks like expert (logit >> 0).

        Returns: (B,)
        """
        with torch.no_grad():
            logit = self.forward(amp_obs)                       # (B, 1)
            r = reward_scale * torch.sigmoid(logit).squeeze(-1)  # (B,) ∈ [0, reward_scale]
        return r

    def compute_loss(
        self,
        expert_obs: Tensor,
        agent_obs: Tensor,
        grad_penalty_coef: float = 10.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total discriminator loss.

        Loss = BCE(expert=real, agent=fake) + gradient_penalty

        Returns:
            loss:  scalar tensor (with grad)
            info:  dict with loss components for logging
        """
        # Expert: label 1, Agent: label 0
        logit_expert = self.forward(expert_obs)
        logit_agent  = self.forward(agent_obs)

        loss_expert = nn.functional.binary_cross_entropy_with_logits(
            logit_expert, torch.ones_like(logit_expert)
        )
        loss_agent = nn.functional.binary_cross_entropy_with_logits(
            logit_agent, torch.zeros_like(logit_agent)
        )
        loss_disc = loss_expert + loss_agent

        # Gradient penalty (WGAN-GP style, interpolated)
        grad_pen = self._gradient_penalty(expert_obs, agent_obs)
        loss = loss_disc + grad_penalty_coef * grad_pen

        info = {
            "disc/loss_expert": loss_expert.item(),
            "disc/loss_agent":  loss_agent.item(),
            "disc/grad_pen":    grad_pen.item(),
            "disc/loss_total":  loss.item(),
            "disc/acc_expert":  (logit_expert > 0).float().mean().item(),
            "disc/acc_agent":   (logit_agent  < 0).float().mean().item(),
        }
        return loss, info

    def _gradient_penalty(self, expert_obs: Tensor, agent_obs: Tensor) -> Tensor:
        """WGAN-GP gradient penalty between expert and agent samples."""
        B = min(expert_obs.shape[0], agent_obs.shape[0])
        expert_obs = expert_obs[:B]
        agent_obs  = agent_obs[:B]

        alpha = torch.rand(B, 1, device=expert_obs.device)
        interp = (alpha * expert_obs + (1.0 - alpha) * agent_obs).requires_grad_(True)
        out = self.forward(interp).sum()
        grad = autograd.grad(
            outputs=out,
            inputs=interp,
            create_graph=True,
        )[0]                                      # (B, amp_obs_dim)
        grad_norm = grad.norm(2, dim=1)           # (B,)
        penalty = (grad_norm - 1.0).pow(2).mean()
        return penalty
