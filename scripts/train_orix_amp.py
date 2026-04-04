#!/usr/bin/env python3
"""Train Orix Dog AMP with amp-rsl-rl + Isaac Lab Manager-based env.

Usage:
    cd <repo_root>
    python scripts/train_orix_amp.py --num_envs 4096 --headless --max_iterations 10000

Note: requires converting raw A1 motions first:
    python motions/convert_to_amp_rsl_rl.py <a1_raw_dir>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Orix AMP")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post Isaac Sim init imports ──
import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register our env
from orix_amp_manager_cfg import OrixAmpRoughEnvCfg

gym.register(
    id="OrixAMP-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": OrixAmpRoughEnvCfg},
)

# amp-rsl-rl
from amp_rsl_rl.runners import AMPOnPolicyRunner


def main():
    # ── Create env ──
    env_cfg = OrixAmpRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = gym.make("OrixAMP-v1", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── AMP training config ──
    motion_data_dir = os.path.join(os.path.dirname(__file__), "..", "motions")

    train_cfg = {
        # Runner
        "num_steps_per_env": 24,
        "save_interval": 500,
        "empirical_normalization": False,
        "logger": "tensorboard",

        # Policy (ActorCritic)
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },

        # Algorithm (AMP_PPO)
        "algorithm": {
            "class_name": "AMP_PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            # AMP specific
            "amp_replay_buffer_size": 1000000,
            "min_std": 0.2,
            "task_reward_lerp": 0.3,  # 30% task + 70% style
        },

        # Discriminator
        "discriminator": {
            "hidden_dims": [1024, 512],
            "reward_scale": 2.0,
        },

        # AMP motion data
        "amp_data_path": motion_data_dir,
        # Names match .npy files produced by motions/convert_to_amp_rsl_rl.py
        "dataset_names": ["orix_amp_trot1", "orix_amp_trot2", "orix_amp_hop1", "orix_amp_hop2"],
        "dataset_weights": [1.0, 1.0, 0.5, 0.5],  # trot weighted more
        "slow_down_factor": 1,
    }

    # ── Log directory ──
    log_dir = os.path.join("logs", "orix_amp_v1", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # ── Create runner ──
    runner = AMPOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=env.unwrapped.device,
    )

    if args.resume:
        print(f"[Resume] {args.resume}")
        runner.load(args.resume)

    # ── Train ──
    print(f"=== Orix AMP Training ===")
    print(f"  Envs: {args.num_envs}")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  Motion data: {motion_data_dir}")
    print(f"  Task/Style lerp: 0.3 / 0.7")
    print(f"  Log: {log_dir}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
