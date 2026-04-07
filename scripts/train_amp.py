#!/usr/bin/env python3
# Copyright (c) 2026 Inovxio
"""Orix Dog training — uses robot_lab ManagerBasedRLEnv.

Usage:
    # Standard PPO (flat terrain)
    python scripts/train_amp.py --task RobotLab-Isaac-Velocity-Flat-OrixDog-v0 --num_envs 4096 --headless

    # Rough terrain
    python scripts/train_amp.py --task RobotLab-Isaac-Velocity-Rough-OrixDog-v0 --num_envs 4096 --headless

Requires robot_lab to be installed or on PYTHONPATH.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Orix Dog with robot_lab env")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-OrixDog-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ──────────────────────────────────────────────────────
import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: register robot_lab envs
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner


def main():
    torch.manual_seed(args.seed)

    # Create env
    env = gym.make(args.task, cfg=None if args.num_envs is None else None)
    # gym.make with robot_lab auto-configures the env via entry_point
    # Override num_envs if specified
    if args.num_envs:
        env_cfg = env.unwrapped.cfg
        env_cfg.scene.num_envs = args.num_envs
        env.close()
        env = gym.make(args.task, cfg=env_cfg)

    env = RslRlVecEnvWrapper(env)

    # Load agent config from task registry
    agent_cfg_entry = gym.spec(args.task).kwargs.get("rsl_rl_cfg_entry_point")
    if agent_cfg_entry:
        from importlib import import_module
        mod_name, cls_name = agent_cfg_entry.rsplit(":", 1)
        agent_cfg = getattr(import_module(mod_name), cls_name)()
    else:
        # Fallback default
        agent_cfg = RslRlOnPolicyRunnerCfg()

    agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        os.path.dirname(__file__), "..", "logs", "rsl_rl",
        args.task.replace("-", "_"),
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=log_dir,
        device=str(env.unwrapped.device),
    )

    if args.resume:
        runner.load(args.resume)

    print(f"\n=== Orix Dog Training ===")
    print(f"  Task: {args.task}")
    print(f"  Envs: {env.num_envs}  Iter: {args.max_iterations}")
    print(f"  Log:  {log_dir}\n")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
