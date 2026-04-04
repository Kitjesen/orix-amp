#!/usr/bin/env python3
"""Train Orix Dog with AMP — robot_lab env + TienKung AMP runner."""
import argparse
import os
import sys
import glob
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Orix AMP")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--resume", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Insert our rsl_rl (TienKung fork with AMP) before system one ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
our_rsl_rl = os.path.join(PROJECT_DIR, "rsl_rl")

sys.path = [p for p in sys.path if "atom01_train/rsl_rl" not in p]
sys.path.insert(0, our_rsl_rl)
for k in list(sys.modules.keys()):
    if k == "rsl_rl" or k.startswith("rsl_rl."):
        del sys.modules[k]

from rsl_rl.runners import AmpOnPolicyRunner
print("[OK] AmpOnPolicyRunner imported from TienKung rsl_rl")

import robot_lab.tasks  # noqa: register envs
import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

# Import the Isaac Lab vec env wrapper base
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class OrixAmpVecEnvWrapper(RslRlVecEnvWrapper):
    """Wrapper adding AMP obs + TienKung-compatible get_observations."""

    # Leg joint names in Isaac order — used to index robot data by name, not position
    _LEG_JOINTS = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    def __init__(self, env):
        super().__init__(env)
        robot = self.unwrapped.scene["robot"]

        # Cache foot body indices
        self._amp_foot_ids = [robot.data.body_names.index(n)
                              for n in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]
        print(f"[OrixAMP] foot_ids={self._amp_foot_ids}")

        # Cache leg joint indices (robust to extra DOFs like foot_joints)
        self._leg_joint_ids = [robot.data.joint_names.index(n) for n in self._LEG_JOINTS]

    @property
    def step_dt(self):
        return self.unwrapped.step_dt

    def get_observations(self):
        """Override: return (obs_tensor, extras) for TienKung AMP runner."""
        # Get obs from Isaac Lab observation manager
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        # Extract policy obs as flat tensor
        if "policy" in obs_dict:
            parts = obs_dict["policy"]
            if isinstance(parts, dict):
                obs = torch.cat(list(parts.values()), dim=-1)
            else:
                obs = parts
        else:
            obs = torch.cat(list(obs_dict.values()), dim=-1)

        # Build extras with critic obs
        extras = {"observations": {}}
        if "critic" in obs_dict:
            parts = obs_dict["critic"]
            if isinstance(parts, dict):
                extras["observations"]["critic"] = torch.cat(list(parts.values()), dim=-1)
            else:
                extras["observations"]["critic"] = parts

        return obs, extras

    def step(self, actions):
        """Override: return (obs, rewards, dones, infos) for TienKung runner."""
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Flatten policy obs
        if "policy" in obs_dict:
            parts = obs_dict["policy"]
            if isinstance(parts, dict):
                obs = torch.cat(list(parts.values()), dim=-1)
            else:
                obs = parts
        else:
            obs = torch.cat(list(obs_dict.values()), dim=-1)

        dones = terminated | truncated

        return obs, rew, dones, extras

    def get_amp_obs_for_expert_trans(self):
        """AMP obs: joint_pos(12) + joint_vel(12) + foot_pos_local(12) = 36D."""
        robot = self.unwrapped.scene["robot"]
        # Index by cached joint ids — safe even when foot_joints add extra DOFs
        joint_pos = robot.data.joint_pos[:, self._leg_joint_ids]
        joint_vel = robot.data.joint_vel[:, self._leg_joint_ids]
        foot_pos_w = robot.data.body_pos_w[:, self._amp_foot_ids]
        root_pos = robot.data.root_pos_w.unsqueeze(1)
        root_quat_conj = quat_conjugate(robot.data.root_quat_w)
        foot_rel = foot_pos_w - root_pos
        foot_local = torch.zeros_like(foot_rel)
        for i in range(4):
            foot_local[:, i] = quat_apply(root_quat_conj, foot_rel[:, i])
        return torch.cat([joint_pos, joint_vel, foot_local.reshape(-1, 12)], dim=1)


def main():
    from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.orix_dog.flat_env_cfg import OrixDogFlatEnvCfg

    env_cfg = OrixDogFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make("RobotLab-Isaac-Velocity-Flat-OrixDog-v0", cfg=env_cfg)
    env = OrixAmpVecEnvWrapper(env)

    motion_dir = os.path.join(PROJECT_DIR, "motions")
    motion_files = sorted(glob.glob(os.path.join(motion_dir, "orix_amp_*.txt")))
    print(f"[AMP] {len(motion_files)} motion files")

    train_cfg = {
        "runner_class_name": "AmpOnPolicyRunner",
        "experiment_name": "orix_dog_amp",
        "run_name": "",
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": 200,
        "empirical_normalization": False,
        "seed": 42,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "AMPPPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "amp_motion_files": motion_files,
        "amp_num_preload_transitions": 2000000,
        "amp_reward_coef": 2.0,
        "amp_task_reward_lerp": 0.3,
        "amp_discr_hidden_dims": [1024, 512],
        "resume": args.resume is not None,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
    }

    log_dir = os.path.join(PROJECT_DIR, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    device = env.unwrapped.device
    runner = AmpOnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)

    if args.resume:
        runner.load(args.resume)

    print(f"\n=== Orix Dog AMP Training ===")
    print(f"  Envs: {env.num_envs}, Iter: {args.max_iterations}")
    print(f"  AMP: {len(motion_files)} motions, lerp=0.3\n")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
