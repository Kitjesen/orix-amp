#!/usr/bin/env python3
# Copyright (c) 2026 Inovxio
"""Record a video of trained Orix Dog AMP policy using Isaac Lab's built-in video recorder.

Usage:
    python scripts/record_video.py \
        --checkpoint logs/orix_amp/2026-.../model_final.pt \
        --video --video_length 500 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",    type=str, required=True)
parser.add_argument("--num_steps",     type=int, default=500)
parser.add_argument("--cmd_vx",        type=float, default=0.5, help="Forward velocity command")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.orix_amp_env import OrixAmpEnv
from envs.orix_amp_env_cfg import OrixAmpEnvCfg
from amp.networks import ActorCritic
from amp.trainer import AMPConfig

from isaaclab.envs.utils.wrappers.record_video import RecordVideoWrapper


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_saved: AMPConfig = ckpt["cfg"]

    # Build env
    env_cfg = OrixAmpEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.reset_strategy = "default"
    env_cfg.cmd_lin_vel_x_range = (args.cmd_vx, args.cmd_vx)
    env_cfg.cmd_lin_vel_y_range = (0.0, 0.0)
    env_cfg.cmd_ang_vel_z_range = (0.0, 0.0)

    env = OrixAmpEnv(cfg=env_cfg, render_mode="rgb_array")

    # Wrap with video recorder
    log_dir = os.path.dirname(args.checkpoint)
    env = RecordVideoWrapper(
        env,
        video_folder=log_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.num_steps,
        name_prefix="orix_amp",
    )

    # Load policy
    actor_critic = ActorCritic(
        obs_dim        = env_cfg.observation_space,
        action_dim     = env_cfg.action_space,
        actor_hidden   = cfg_saved.actor_hidden,
        critic_hidden  = cfg_saved.critic_hidden,
        critic_obs_dim = env_cfg.state_space,
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()

    # Run
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    print(f"Recording {args.num_steps} steps @ cmd_vx={args.cmd_vx} m/s ...")
    for step in range(args.num_steps):
        with torch.no_grad():
            actions, _ = actor_critic.act(obs)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]

    env.close()
    print(f"Video saved to: {log_dir}/")
    simulation_app.close()


if __name__ == "__main__":
    main()
