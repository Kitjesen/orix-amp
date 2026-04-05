#!/usr/bin/env python3
# Copyright (c) 2026 Inovxio
"""Record a video of trained Orix Dog AMP policy.

Usage:
    python scripts/record_video.py --checkpoint logs/orix_amp/2026-.../model_final.pt
    python scripts/record_video.py --checkpoint logs/orix_amp/2026-.../model_final.pt --num_envs 1
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--num_steps",  type=int, default=500)
parser.add_argument("--output",     type=str, default="orix_amp_video.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force rendering mode on
args.headless      = False
args.enable_cameras = True

app_launcher   = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.orix_amp_env import OrixAmpEnv
from envs.orix_amp_env_cfg import OrixAmpEnvCfg
from amp.networks import ActorCritic
from amp.trainer import AMPConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_saved: AMPConfig = ckpt["cfg"]

    # Build env (1 env, render mode)
    env_cfg = OrixAmpEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reset_strategy = "default"   # stable initial pose for demo

    # Fix commands for demo: walk forward
    env_cfg.cmd_lin_vel_x_range = (0.5, 0.5)
    env_cfg.cmd_lin_vel_y_range = (0.0, 0.0)
    env_cfg.cmd_ang_vel_z_range = (0.0, 0.0)

    env = OrixAmpEnv(cfg=env_cfg, render_mode="rgb_array")

    # Restore actor
    actor_critic = ActorCritic(
        obs_dim        = env_cfg.observation_space,
        action_dim     = env_cfg.action_space,
        actor_hidden   = cfg_saved.actor_hidden,
        critic_hidden  = cfg_saved.critic_hidden,
        critic_obs_dim = env_cfg.state_space,
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()

    # Record
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    frames = []
    print(f"Recording {args.num_steps} steps...")
    for step in range(args.num_steps):
        with torch.no_grad():
            actions, _ = actor_critic.act(obs)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]

    env.close()

    if not frames:
        print("No frames captured. Check render_mode.")
        simulation_app.close()
        return

    # Save video
    try:
        import imageio
        out = args.output
        imageio.mimsave(out, frames, fps=50)
        print(f"Saved: {out}  ({len(frames)} frames)")
    except ImportError:
        print("Install imageio: pip install imageio[ffmpeg]")
        print(f"Captured {len(frames)} frames but could not save.")

    simulation_app.close()


if __name__ == "__main__":
    main()
