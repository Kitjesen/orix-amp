# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP environment — based on robot_lab g1_amp.

Quadruped AMP with:
- 12 DOF actions (joint position targets)
- Imitation reward (joint pos/vel + body pos/rot)
- AMP discriminator observation buffer
- Motion reference from .npz trot data
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .orix_amp_env_cfg import OrixAmpEnvCfg

# Add motions to path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motions.motion_loader_quad import QuadMotionLoader


class OrixAmpEnv(DirectRLEnv):
    cfg: OrixAmpEnvCfg

    def __init__(self, cfg: OrixAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action offset and scale
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = dof_upper - dof_lower

        # Load motion reference
        self._motion_loader = QuadMotionLoader(
            motion_file=self.cfg.motion_file, device=self.device
        )

        # Key body indexes (feet for quadruped)
        key_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = []
        for name in key_body_names:
            try:
                self.key_body_indexes.append(self.robot.data.body_names.index(name))
            except ValueError:
                print(f"[WARN] Body '{name}' not found, using base_link")
                self.key_body_indexes.append(self.ref_body_index)

        # Motion DOF and body index mapping
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(
            [n for n in key_body_names if n in self._motion_loader.body_names]
        )

        # AMP observation buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,)
        )
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            device=self.device,
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        progress = (self.episode_length_buf.float() / (self.max_episode_length - 1)).unsqueeze(-1)

        root_pos_rel = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        key_body_pos_rel = (
            self.robot.data.body_pos_w[:, self.key_body_indexes]
            - self.scene.env_origins.unsqueeze(1)
        )

        obs = torch.cat([
            self.robot.data.joint_pos,                                    # 12
            self.robot.data.joint_vel,                                    # 12
            root_pos_rel,                                                 # 3
            self.robot.data.body_quat_w[:, self.ref_body_index],         # 4
            key_body_pos_rel.reshape(self.num_envs, -1),                 # 4*3=12
            progress,                                                     # 1
        ], dim=-1)  # total = 44

        # Update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        with torch.no_grad():
            current_times = (self.episode_length_buf * self.physics_dt).cpu().numpy()
            (
                ref_dof_pos, ref_dof_vel,
                ref_body_pos, ref_body_rot, _, _,
            ) = self._motion_loader.sample(num_samples=self.num_envs, times=current_times)

            ref_joint_pos = ref_dof_pos[:, self.motion_dof_indexes]
            ref_joint_vel = ref_dof_vel[:, self.motion_dof_indexes]
            ref_root_pos = ref_body_pos[:, self.motion_ref_body_index]
            ref_root_quat = ref_body_rot[:, self.motion_ref_body_index]

        # 1. Joint position imitation
        joint_pos_err = torch.square(self.robot.data.joint_pos - ref_joint_pos).sum(dim=-1)
        rew_joint_pos = self.cfg.rew_imitation_joint_pos * torch.exp(
            -joint_pos_err / self.cfg.imitation_sigma_joint_pos
        )

        # 2. Joint velocity imitation
        joint_vel_err = torch.square(self.robot.data.joint_vel - ref_joint_vel).sum(dim=-1)
        rew_joint_vel = self.cfg.rew_imitation_joint_vel * torch.exp(
            -joint_vel_err / self.cfg.imitation_sigma_joint_vel
        )

        # 3. Root position imitation
        root_pos_rel = self.robot.data.body_pos_w[:, self.ref_body_index] - self.scene.env_origins
        pos_err = torch.square(root_pos_rel - ref_root_pos).sum(dim=-1)
        rew_pos = self.cfg.rew_imitation_pos * torch.exp(-pos_err / self.cfg.imitation_sigma_pos)

        # 4. Root rotation imitation
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        rot_err = torch.square(root_quat - ref_root_quat).sum(dim=-1)
        rew_rot = self.cfg.rew_imitation_rot * torch.exp(-rot_err / self.cfg.imitation_sigma_rot)

        # 5. Regularization
        rew_action = self.cfg.rew_action_l2 * torch.square(self.actions).sum(dim=-1)
        rew_joint_acc = self.cfg.rew_joint_acc_l2 * torch.square(
            self.robot.data.joint_acc[:self.cfg.action_space] if hasattr(self.robot.data, 'joint_acc')
            else torch.zeros(self.num_envs, device=self.device)
        ).sum(dim=-1) if hasattr(self.robot.data, 'joint_acc') else torch.zeros(self.num_envs, device=self.device)

        total = rew_joint_pos + rew_joint_vel + rew_pos + rew_rot + rew_action

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.early_termination:
            base_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            terminated = base_height < self.cfg.termination_height

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.robot.reset(env_ids)

        if self.cfg.reset_strategy == "random-start":
            # Reset to motion start with slight randomization
            times = np.zeros(len(env_ids))
            (dof_pos, dof_vel, body_pos, body_rot, _, _) = self._motion_loader.sample(
                num_samples=len(env_ids), times=times
            )

            joint_pos = dof_pos[:, self.motion_dof_indexes]
            joint_vel = dof_vel[:, self.motion_dof_indexes]
            # Add noise
            joint_pos += torch.randn_like(joint_pos) * 0.05
            joint_vel += torch.randn_like(joint_vel) * 0.1

            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            root_pos = body_pos[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
            root_quat = body_rot[:, self.motion_ref_body_index]

            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] = root_pos
            root_state[:, 3:7] = root_quat
            self.robot.write_root_state_to_sim(root_state, env_ids)
        else:
            # Default reset
            joint_pos = self.robot.data.default_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            root_state = self.robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += self.scene.env_origins[env_ids]
            self.robot.write_root_state_to_sim(root_state, env_ids)

        # Reset AMP buffer
        self.amp_observation_buffer[env_ids] = 0.0
