# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""AMP-PPO Trainer.

Self-contained: no robot_lab, no amp-rsl-rl, no TienKung fork.
Depends only on: isaaclab, torch, our own networks.py and buffer.py.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .networks import ActorCritic, AMPDiscriminator
from .buffer import RolloutBuffer, AMPReplayBuffer


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AMPConfig:
    # PPO
    num_steps_per_env:    int   = 24
    num_learning_epochs:  int   = 5
    num_mini_batches:     int   = 4
    clip_param:           float = 0.2
    value_loss_coef:      float = 1.0
    entropy_coef:         float = 0.01
    learning_rate:        float = 1e-3
    max_grad_norm:        float = 1.0
    gamma:                float = 0.99
    lam:                  float = 0.95
    desired_kl:           float = 0.01
    use_clipped_value_loss: bool = True

    # AMP
    task_reward_lerp:      float = 0.3     # w_task; style weight = 1 - w_task
    amp_replay_buffer_size: int  = 1_000_000
    amp_expert_preload:    int   = 200_000  # expert transitions to pre-fill
    amp_batch_size:        int   = 512
    num_disc_updates:      int   = 5       # discriminator gradient steps per iteration (match PPO epochs)
    disc_grad_penalty:     float = 10.0
    disc_reward_scale:     float = 2.0
    disc_learning_rate:    float = 1e-4

    # Network
    actor_hidden:  list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    disc_hidden:   list[int] = field(default_factory=lambda: [1024, 512])
    init_noise_std: float = 1.0

    # I/O
    save_interval: int = 500
    log_dir:       str = "logs/orix_amp"


# ── Trainer ───────────────────────────────────────────────────────────────────

class AMPTrainer:
    """Trains an AMP policy with PPO and an adversarial discriminator.

    Algorithm:
        for each iteration:
            1. Collect rollout (policy steps)
            2. Compute discriminator style reward
            3. Blend: r_total = w_task*r_task + (1-w_task)*r_style
            4. GAE returns
            5. Update actor-critic with PPO
            6. Update discriminator with BCE + gradient penalty
            7. Log and checkpoint
    """

    def __init__(
        self,
        env,
        cfg: AMPConfig,
        device: torch.device,
    ) -> None:
        self.env    = env
        self.cfg    = cfg
        self.device = device

        # Dimensions from env
        self.obs_dim      = env.cfg.observation_space   # 41 (actor)
        self.critic_dim   = env.cfg.state_space or env.cfg.observation_space  # 73 (privileged)
        self.action_dim   = env.cfg.action_space        # 12
        self.amp_obs_dim  = env.amp_observation_size    # 3 * 40 = 120
        self.num_envs     = env.num_envs

        # Networks
        self.actor_critic = ActorCritic(
            obs_dim        = self.obs_dim,
            action_dim     = self.action_dim,
            actor_hidden   = cfg.actor_hidden,
            critic_hidden  = cfg.critic_hidden,
            init_noise_std = cfg.init_noise_std,
            critic_obs_dim = self.critic_dim,
        ).to(device)

        self.discriminator = AMPDiscriminator(
            amp_obs_dim = self.amp_obs_dim,
            hidden_dims = cfg.disc_hidden,
        ).to(device)

        # Optimizers (separate, per standard AMP practice)
        self.optimizer_ac   = torch.optim.Adam(
            self.actor_critic.parameters(), lr=cfg.learning_rate
        )
        self.optimizer_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=cfg.disc_learning_rate
        )

        # Replay buffers
        self.expert_buffer = AMPReplayBuffer(cfg.amp_replay_buffer_size, self.amp_obs_dim, device)
        self.agent_buffer  = AMPReplayBuffer(cfg.amp_replay_buffer_size, self.amp_obs_dim, device)

        # Pre-fill expert buffer from motion reference
        self._prefill_expert_buffer(cfg.amp_expert_preload)

        # State tracking
        self._current_obs: Tensor | None = None
        self.it = 0
        os.makedirs(cfg.log_dir, exist_ok=True)

    # ── Expert buffer ─────────────────────────────────────────────────────────

    def _prefill_expert_buffer(self, total: int) -> None:
        """Fill expert buffer with reference motion samples.

        collect_reference_motions() returns single-frame (B, 43) tensors.
        We stack 3 independent random frames to match the env's 3-frame
        stacked AMP obs format (B, 129).  This is the standard AMP approximation
        when sequential data isn't pre-indexed.
        """
        print(f"[AMP] Pre-filling expert buffer with {total} samples...")
        batch = 4096
        filled = 0
        while filled < total:
            n = min(batch, total - filled)
            frames = [self.env.collect_reference_motions(n) for _ in range(3)]
            stacked = torch.cat(frames, dim=-1)   # (n, 129)
            self.expert_buffer.add(stacked)
            filled += n
        print(f"[AMP] Expert buffer ready: {len(self.expert_buffer)} samples")

    # ── Rollout ───────────────────────────────────────────────────────────────

    def _collect_rollout(self) -> RolloutBuffer:
        rollout = RolloutBuffer(
            num_envs       = self.num_envs,
            num_steps      = self.cfg.num_steps_per_env,
            obs_dim        = self.obs_dim,
            action_dim     = self.action_dim,
            amp_obs_dim    = self.amp_obs_dim,
            device         = self.device,
            critic_obs_dim = self.critic_dim,
        )

        # Use cached obs if available (continuous across iterations)
        if self._current_obs is None:
            obs_dict, _ = self.env.reset()
            obs        = obs_dict["policy"]
            critic_obs = obs_dict.get("critic", obs)
        else:
            obs, critic_obs = self._current_obs

        self.actor_critic.eval()
        for step in range(self.cfg.num_steps_per_env):
            actions, log_probs = self.actor_critic.act(obs)
            values             = self.actor_critic.get_value(critic_obs)

            obs_dict, task_rew, terminated, truncated, extras = self.env.step(actions)
            amp_obs = extras["amp_obs"]

            style_rew = self.discriminator.compute_style_reward(amp_obs, self.cfg.disc_reward_scale)
            total_rew = self._blend_reward(task_rew, style_rew)

            rollout.add(step, obs, actions, log_probs, total_rew, values, terminated, amp_obs, critic_obs)
            self.agent_buffer.add(amp_obs)

            obs        = obs_dict["policy"]
            critic_obs = obs_dict.get("critic", obs)

        self._current_obs = (obs, critic_obs)

        # Bootstrap value from privileged critic obs
        last_val = self.actor_critic.get_value(critic_obs)
        rollout.compute_returns(last_val, self.cfg.gamma, self.cfg.lam)
        return rollout

    def _blend_reward(self, task_rew: Tensor, style_rew: Tensor) -> Tensor:
        w = self.cfg.task_reward_lerp
        return w * task_rew + (1.0 - w) * style_rew

    # ── PPO update ────────────────────────────────────────────────────────────

    def _update_ppo(self, rollout: RolloutBuffer) -> dict[str, float]:
        self.actor_critic.train()

        stats: dict[str, list[float]] = {
            "ppo/policy_loss": [], "ppo/value_loss": [],
            "ppo/entropy":     [], "ppo/kl":         [],
        }

        for _ in range(self.cfg.num_learning_epochs):
            for mb in rollout.get_minibatches(self.cfg.num_mini_batches):
                obs_mb      = mb["obs"]
                actions_mb  = mb["actions"]
                lp_old_mb   = mb["log_probs_old"]
                ret_mb      = mb["returns"]
                adv_mb      = mb["advantages"]

                # Normalize advantages per mini-batch
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                values, log_probs, entropy = self.actor_critic.evaluate(
                    obs_mb, actions_mb, critic_obs=mb["critic_obs"]
                )

                # Policy loss (PPO-clip)
                ratio   = (log_probs - lp_old_mb).exp()
                surr1   = ratio * adv_mb
                surr2   = ratio.clamp(1 - self.cfg.clip_param, 1 + self.cfg.clip_param) * adv_mb
                pol_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.cfg.use_clipped_value_loss:
                    val_old_mb = mb["values_old"]
                    v_clipped = val_old_mb + (values - val_old_mb).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                    val_loss  = torch.max(
                        (values - ret_mb).pow(2),
                        (v_clipped - ret_mb).pow(2)
                    ).mean()
                else:
                    val_loss = (values - ret_mb).pow(2).mean()

                loss = (
                    pol_loss
                    + self.cfg.value_loss_coef * val_loss
                    - self.cfg.entropy_coef * entropy.mean()
                )

                self.optimizer_ac.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer_ac.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (log_probs - lp_old_mb)).mean()

                stats["ppo/policy_loss"].append(pol_loss.item())
                stats["ppo/value_loss"].append(val_loss.item())
                stats["ppo/entropy"].append(entropy.mean().item())
                stats["ppo/kl"].append(approx_kl.item())

        # Adaptive learning rate
        mean_kl = sum(stats["ppo/kl"]) / len(stats["ppo/kl"])
        if mean_kl > 1.5 * self.cfg.desired_kl:
            for g in self.optimizer_ac.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-6)
        elif mean_kl < self.cfg.desired_kl / 1.5:
            for g in self.optimizer_ac.param_groups:
                g["lr"] = min(g["lr"] * 2.0, 1e-2)

        return {k: sum(v) / len(v) for k, v in stats.items()}

    # ── Discriminator update ──────────────────────────────────────────────────

    def _update_discriminator(self) -> dict[str, float]:
        """Multiple discriminator gradient steps per iteration.

        num_disc_updates steps to keep pace with PPO's num_learning_epochs updates.
        Returns stats from the final update step.
        """
        B = self.cfg.amp_batch_size
        self.discriminator.train()
        info: dict[str, float] = {}

        for _ in range(self.cfg.num_disc_updates):
            expert_obs = self.expert_buffer.sample(B)
            agent_obs  = self.agent_buffer.sample(B)

            self.optimizer_disc.zero_grad()
            loss, info = self.discriminator.compute_loss(
                expert_obs, agent_obs, self.cfg.disc_grad_penalty
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
            self.optimizer_disc.step()

        return info

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, num_iterations: int) -> None:
        print(f"\n{'='*60}")
        print(f"  Orix AMP Training (self-contained)")
        print(f"  Envs: {self.num_envs}   Steps/iter: {self.cfg.num_steps_per_env}")
        print(f"  Iters: {num_iterations}   Log: {self.cfg.log_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for it in range(self.it, self.it + num_iterations):
            iter_start = time.time()

            # 1. Collect rollout
            rollout = self._collect_rollout()

            # 2. Update policy
            ppo_stats = self._update_ppo(rollout)

            # 3. Update discriminator
            disc_stats = self._update_discriminator()

            self.it = it + 1

            # 4. Logging
            iter_time = time.time() - iter_start
            fps = (self.num_envs * self.cfg.num_steps_per_env) / iter_time

            mean_rew = rollout.rewards.mean().item()
            mean_ret = rollout.returns.mean().item() if rollout.returns is not None else 0.0

            if (it + 1) % 10 == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer_ac.param_groups[0]["lr"]
                print(
                    f"[{it+1:5d}/{num_iterations}] "
                    f"rew={mean_rew:.3f}  ret={mean_ret:.3f}  "
                    f"pol={ppo_stats['ppo/policy_loss']:.4f}  "
                    f"val={ppo_stats['ppo/value_loss']:.4f}  "
                    f"disc={disc_stats['disc/loss_total']:.4f}  "
                    f"acc_r={disc_stats['disc/acc_expert']:.2f}/"
                    f"{disc_stats['disc/acc_agent']:.2f}  "
                    f"fps={fps:.0f}  lr={lr:.2e}  t={elapsed:.0f}s"
                )

            # 5. Checkpoint
            if (it + 1) % self.cfg.save_interval == 0:
                self.save(os.path.join(self.cfg.log_dir, f"model_{it+1}.pt"))

        self.save(os.path.join(self.cfg.log_dir, "model_final.pt"))
        print(f"\nTraining complete. Total time: {time.time()-start_time:.0f}s")

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "iteration":       self.it,
            "actor_critic":    self.actor_critic.state_dict(),
            "discriminator":   self.discriminator.state_dict(),
            "optimizer_ac":    self.optimizer_ac.state_dict(),
            "optimizer_disc":  self.optimizer_disc.state_dict(),
            "cfg":             self.cfg,
        }, path)
        print(f"[AMP] Saved: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.it = ckpt["iteration"]
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.optimizer_ac.load_state_dict(ckpt["optimizer_ac"])
        self.optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
        print(f"[AMP] Loaded: {path} (iteration {self.it})")
