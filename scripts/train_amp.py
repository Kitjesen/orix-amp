#!/usr/bin/env python3
# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP Training — self-contained, no external RL libraries.

Dependencies: isaaclab, torch, gymnasium only.

Usage:
    python scripts/train_amp.py --num_envs 4096 --headless
    python scripts/train_amp.py --num_envs 4096 --headless --resume logs/orix_amp/model_500.pt
"""
import argparse
import os
import sys

# ── Isaac Sim must be launched BEFORE any isaaclab imports ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Orix Dog AMP — self-contained training")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=10000)
parser.add_argument("--resume",         type=str,   default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--log_dir",        type=str,   default=None,
                    help="Override log directory")
parser.add_argument("--motion_file",    type=str,   default=None,
                    help="Override motion .npz file path")
parser.add_argument("--task_lerp",      type=float, default=0.3,
                    help="Task reward weight (0=pure style, 1=pure task)")
parser.add_argument("--seed",           type=int,   default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ──
import torch
from datetime import datetime

# Add repo root to path so `envs` and `amp` packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.orix_amp_env import OrixAmpEnv
from envs.orix_amp_env_cfg import OrixAmpEnvCfg
from amp.trainer import AMPTrainer, AMPConfig


def main() -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build env config ──
    env_cfg = OrixAmpEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.motion_file:
        env_cfg.motion_file = args.motion_file

    # ── Create env ──
    env = OrixAmpEnv(cfg=env_cfg)

    # ── Build training config ──
    log_dir = args.log_dir or os.path.join(
        os.path.dirname(__file__), "..", "logs", "orix_amp",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    cfg = AMPConfig(
        # PPO
        num_steps_per_env    = 24,
        num_learning_epochs  = 5,
        num_mini_batches     = 4,
        clip_param           = 0.2,
        value_loss_coef      = 1.0,
        entropy_coef         = 0.01,
        learning_rate        = 1e-3,
        max_grad_norm        = 1.0,
        gamma                = 0.99,
        lam                  = 0.95,
        desired_kl           = 0.01,
        use_clipped_value_loss = True,
        # AMP
        task_reward_lerp     = args.task_lerp,
        amp_replay_buffer_size = 1_000_000,
        amp_expert_preload   = 200_000,
        amp_batch_size       = 512,
        disc_grad_penalty    = 10.0,
        disc_reward_scale    = 2.0,
        disc_learning_rate   = 1e-4,
        # Networks
        actor_hidden  = [512, 256, 128],
        critic_hidden = [512, 256, 128],
        disc_hidden   = [1024, 512],
        init_noise_std = 1.0,
        # I/O
        save_interval = 500,
        log_dir       = log_dir,
    )

    # ── Create trainer ──
    trainer = AMPTrainer(env=env, cfg=cfg, device=device)

    if args.resume:
        trainer.load(args.resume)

    # ── Train ──
    trainer.train(num_iterations=args.max_iterations)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
