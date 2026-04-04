#!/usr/bin/env python3
"""Orix Dog AMP training — TienKung rsl_rl + robot_lab env.

Fixes vs v2:
  - AMP foot bodies: FL/FR/RL/RR_calf (merge_fixed_joints merges _foot into _calf)
  - amp_task_reward_lerp: 0.3 → 0.5  (balanced task vs style)
  - amp_reward_coef: 2.0 → 1.0
  - entropy_coef: 0.01 → 0.001       (let noise_std collapse)
  - init_noise_std: 1.0 → 0.3
  - action_rate_l2: -0.01 → -0.5    (smooth actions)
  - joint_pos_limits: -2.0 → -5.0   (hard limit enforcement)
  - init pose thigh: ±0.8 → ±0.65   (more margin from limits)
"""
import argparse
import os
import sys
import glob
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Orix AMP v3")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=15000)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint dir to resume from")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Load TienKung rsl_rl (has AmpOnPolicyRunner) ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
_tienkung_path = os.path.join(PROJECT_DIR, "rsl_rl")
if not os.path.isdir(_tienkung_path):
    raise RuntimeError(
        f"TienKung rsl_rl not found at {_tienkung_path}.\n"
        "Run: ln -s ~/hongsenpang/RLbased/orix_amp_rslrl/rsl_rl <project>/rsl_rl"
    )
sys.path = [p for p in sys.path if "rsl_rl" not in p]
sys.path.insert(0, _tienkung_path)
for k in list(sys.modules.keys()):
    if k == "rsl_rl" or k.startswith("rsl_rl."):
        del sys.modules[k]
from rsl_rl.runners import AmpOnPolicyRunner  # noqa: E402
print("[OK] AmpOnPolicyRunner from TienKung rsl_rl")

import robot_lab.tasks  # noqa: E402  register robot_lab envs
import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


# ── AMP wrapper ───────────────────────────────────────────────────────────────
class OrixAmpVecEnvWrapper(RslRlVecEnvWrapper):
    """Adds AMP obs + TienKung-compatible interface to robot_lab env.

    AMP obs (36D): joint_pos(12) + joint_vel(12) + foot_pos_local(12)
    Foot bodies: *_calf  (merge_fixed_joints folds *_foot into *_calf)
    Joint order (FL, FR, RL, RR) matches orix_amp_*.txt expert data.
    """

    _FOOT_BODIES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    _LEG_JOINTS = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    def __init__(self, env):
        super().__init__(env)
        robot = self.unwrapped.scene["robot"]

        self._foot_ids = [robot.data.body_names.index(n) for n in self._FOOT_BODIES]
        self._joint_ids = [robot.data.joint_names.index(n) for n in self._LEG_JOINTS]
        self._reset_ids = torch.tensor([], dtype=torch.long,
                                       device=self.unwrapped.device)
        print(f"[OrixAMP v3] foot_ids={self._foot_ids}  joint_ids={self._joint_ids[:3]}...")

    @property
    def step_dt(self):
        return self.unwrapped.step_dt

    @property
    def reset_env_ids(self):
        return self._reset_ids

    # ── TienKung interface ────────────────────────────────────────────────────
    def get_observations(self):
        obs_dict = self.unwrapped.observation_manager.compute()
        obs = self._flatten_policy(obs_dict)
        extras = {"observations": {}}
        if "critic" in obs_dict:
            extras["observations"]["critic"] = self._flatten(obs_dict["critic"])
        return obs, extras

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = self._flatten_policy(obs_dict)
        dones = terminated | truncated
        self._reset_ids = dones.nonzero(as_tuple=False).flatten()
        if "observations" not in extras:
            extras["observations"] = {}
        if "critic" in obs_dict:
            extras["observations"]["critic"] = self._flatten(obs_dict["critic"])
        return obs, rew, dones, extras

    def get_amp_obs_for_expert_trans(self):
        """36D AMP obs: joint_pos(12) + joint_vel(12) + foot_pos_local(12)."""
        robot = self.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[:, self._joint_ids]
        joint_vel = robot.data.joint_vel[:, self._joint_ids]

        foot_pos_w = robot.data.body_pos_w[:, self._foot_ids]       # [N, 4, 3]
        root_pos_w = robot.data.root_pos_w.unsqueeze(1)              # [N, 1, 3]
        root_quat_conj = quat_conjugate(robot.data.root_quat_w)      # [N, 4]
        foot_rel = foot_pos_w - root_pos_w                           # [N, 4, 3]
        foot_local = torch.stack(
            [quat_apply(root_quat_conj, foot_rel[:, i]) for i in range(4)], dim=1
        )                                                             # [N, 4, 3]
        return torch.cat([joint_pos, joint_vel, foot_local.reshape(-1, 12)], dim=1)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _flatten(x):
        if isinstance(x, dict):
            return torch.cat(list(x.values()), dim=-1)
        return x

    def _flatten_policy(self, obs_dict):
        key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
        return self._flatten(obs_dict[key])


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # Register orix env IDs from repo config/
    sys.path.insert(0, PROJECT_DIR)
    import config  # noqa: triggers gym.register for OrixAmp-* envs

    from config.flat_env_cfg import OrixDogFlatEnvCfg

    env_cfg = OrixDogFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make("OrixAmp-Isaac-Velocity-Flat-v0", cfg=env_cfg)
    env = OrixAmpVecEnvWrapper(env)

    motion_dir = os.path.join(PROJECT_DIR, "motions")
    motion_files = sorted(glob.glob(os.path.join(motion_dir, "orix_amp_*.txt")))
    if not motion_files:
        raise RuntimeError(f"No orix_amp_*.txt found in {motion_dir}")
    print(f"[AMP] {len(motion_files)} motion files: {[os.path.basename(f) for f in motion_files]}")

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
            "init_noise_std": 0.3,          # was 1.0 → faster convergence
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
            "entropy_coef": 0.001,          # was 0.01 → let noise_std collapse
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
        "amp_reward_coef": 1.0,             # was 2.0
        "amp_task_reward_lerp": 0.5,        # was 0.3 → 50/50 task vs style
        "amp_discr_hidden_dims": [512, 256, 128],
        "resume": args.resume is not None,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "min_normalized_std": [0.05] * 12,
    }

    log_dir = os.path.join(PROJECT_DIR, "logs",
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = AmpOnPolicyRunner(
        env=env, train_cfg=train_cfg, log_dir=log_dir,
        device=str(env.unwrapped.device),
    )
    if args.resume:
        runner.load(args.resume)

    print(f"\n=== Orix Dog AMP v3 ===")
    print(f"  Envs: {env.num_envs}  |  Iter: {args.max_iterations}")
    print(f"  AMP lerp=0.5  coef=1.0  entropy=0.001  noise_std=0.3")
    print(f"  action_rate=-0.5  joint_limits=-5.0\n")

    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
