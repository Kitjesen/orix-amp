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

from .orix_amp_env_cfg import OrixAmpEnvCfg

# Add motions to path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motions.motion_loader_quad import QuadMotionLoader


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q (wxyz convention)."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    # v' = q^{-1} v q  (passive rotation: body-frame coords of world vector)
    t = 2.0 * torch.cross(torch.cat([x, y, z], dim=-1), v, dim=-1)
    return v - w * t + torch.cross(torch.cat([x, y, z], dim=-1), t, dim=-1)


class OrixAmpEnv(DirectRLEnv):
    cfg: OrixAmpEnvCfg

    def __init__(self, cfg: OrixAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action offset (default standing pose) and fixed scale
        self.action_offset = self.robot.data.default_joint_pos[0].clone()
        self.action_scale = 0.5  # radians — policy outputs residuals around default pose

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

        root_pos_w   = self.robot.data.body_pos_w[:, self.ref_body_index]   # (N, 3)
        root_quat_w  = self.robot.data.body_quat_w[:, self.ref_body_index]  # (N, 4) wxyz

        # base_height: z above ground (env-relative) — 1D, IMU-estimable
        base_height = (root_pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1)  # (N, 1)

        # projected_gravity: gravity vec in body frame — IMU-measurable (roll/pitch, yaw-invariant)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        projected_gravity = _quat_rotate_inverse(root_quat_w, gravity_w)   # (N, 3)

        # key_body positions relative to root (FK-computable on real robot)
        key_body_pos_root_rel = (
            self.robot.data.body_pos_w[:, self.key_body_indexes]  # (N, 4, 3)
            - root_pos_w.unsqueeze(1)
        ).reshape(self.num_envs, -1)                              # (N, 12)

        # ── AMP obs (40D) — all realizable on real robot ──────────────────────
        amp_obs = torch.cat([
            self.robot.data.joint_pos,   # 12  encoder
            self.robot.data.joint_vel,   # 12  encoder
            base_height,                 #  1  IMU/barometer
            projected_gravity,           #  3  IMU
            key_body_pos_root_rel,       # 12  FK
        ], dim=-1)  # 40

        # ── Actor obs (41D) ───────────────────────────────────────────────────
        policy_obs = torch.cat([amp_obs, progress], dim=-1)  # 41

        # ── Critic obs (73D) — privileged sim-only info ───────────────────────
        base_lin_vel = _quat_rotate_inverse(root_quat_w, self.robot.data.root_lin_vel_w)  # (N, 3)
        # Contact forces on feet: scalar magnitude per foot (4D)
        feet_contact = torch.zeros(self.num_envs, 4, device=self.device)
        if hasattr(self.robot.data, "net_contact_force_w"):
            foot_forces = self.robot.data.net_contact_force_w[:, self.key_body_indexes]  # (N, 4, 3)
            feet_contact = foot_forces.norm(dim=-1)                                      # (N, 4)
        # Height scan: 5×5 grid around robot base (25D)
        height_scan = self._get_height_scan(root_pos_w)  # (N, 25)

        critic_obs = torch.cat([
            policy_obs,      # 41
            base_lin_vel,    #  3  ground truth from sim
            feet_contact,    #  4  contact forces per foot
            height_scan,     # 25  terrain height around robot
        ], dim=-1)  # 73

        # ── AMP history buffer ────────────────────────────────────────────────
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": policy_obs, "critic": critic_obs}

    def _get_height_scan(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """Simple 5×5 height scan grid around robot base (25 points, 0.25m spacing).

        Returns heights relative to robot base z.
        Falls back to zeros if terrain height not queryable.
        """
        N = self.num_envs
        # 5×5 grid offsets: -0.5, -0.25, 0, 0.25, 0.5 m
        offsets = torch.linspace(-0.5, 0.5, 5, device=self.device)
        grid_x, grid_y = torch.meshgrid(offsets, offsets, indexing="ij")  # (5, 5)
        grid_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (25, 2)

        # Sample positions in world frame
        scan_pos_w = root_pos_w[:, :2].unsqueeze(1) + grid_flat.unsqueeze(0)  # (N, 25, 2)

        # Query terrain height — for flat ground: 0.0
        ground_z = torch.zeros(N, 25, device=self.device)
        scan_heights = ground_z - root_pos_w[:, 2:3]  # height relative to robot z (N, 25)
        return scan_heights

    def _get_rewards(self) -> torch.Tensor:
        with torch.no_grad():
            # step_dt = decimation * sim.dt (policy-level time, not physics sub-step time)
            current_times = (self.episode_length_buf * self.step_dt).cpu().numpy()
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

        # 4. Root rotation imitation — use 1 - |dot(q, q_ref)|^2 to handle q == -q symmetry
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        quat_dot = (root_quat * ref_root_quat).sum(dim=-1)
        rot_err = 1.0 - torch.square(quat_dot)
        rew_rot = self.cfg.rew_imitation_rot * torch.exp(-rot_err / self.cfg.imitation_sigma_rot)

        # 5. Regularization
        rew_action = self.cfg.rew_action_l2 * torch.square(self.actions).sum(dim=-1)
        rew_joint_vel_l2 = self.cfg.rew_joint_vel_l2 * torch.square(self.robot.data.joint_vel).sum(dim=-1)

        if hasattr(self.robot.data, 'joint_acc'):
            rew_joint_acc = self.cfg.rew_joint_acc_l2 * torch.square(
                self.robot.data.joint_acc[:, :self.cfg.action_space]
            ).sum(dim=-1)
        else:
            rew_joint_acc = torch.zeros(self.num_envs, device=self.device)

        # 6. Joint position limit penalty
        joint_pos = self.robot.data.joint_pos
        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        out_of_range = ((joint_pos < lower) | (joint_pos > upper)).float().sum(dim=-1)
        rew_joint_limits = self.cfg.rew_joint_pos_limits * out_of_range

        # 7. Termination penalty
        base_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
        rew_termination = self.cfg.rew_termination * (base_height < self.cfg.termination_height).float()

        total = (rew_joint_pos + rew_joint_vel + rew_pos + rew_rot
                 + rew_action + rew_joint_vel_l2 + rew_joint_acc
                 + rew_joint_limits + rew_termination)

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.early_termination:
            base_height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            terminated = base_height < self.cfg.termination_height

        return terminated, time_out

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Sample AMP obs from motion reference data (40D, matches env amp_obs).

        Format: joint_pos(12) + joint_vel(12) + base_height(1) + projected_gravity(3) + key_body_pos(12)
        """
        times = np.random.uniform(0, self._motion_loader.duration, num_samples)
        dof_pos, dof_vel, body_pos, body_rot, _, _ = self._motion_loader.sample(
            num_samples=num_samples, times=times
        )

        ref_joint_pos  = dof_pos[:, self.motion_dof_indexes]           # (N, 12)
        ref_joint_vel  = dof_vel[:, self.motion_dof_indexes]           # (N, 12)
        ref_root_pos   = body_pos[:, self.motion_ref_body_index]       # (N, 3)
        ref_root_quat  = body_rot[:, self.motion_ref_body_index]       # (N, 4) wxyz

        # base_height: z coordinate of root
        base_height = ref_root_pos[:, 2:3]                             # (N, 1)

        # projected_gravity from reference quaternion
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(num_samples, 3)
        projected_gravity = _quat_rotate_inverse(ref_root_quat, gravity_w)  # (N, 3)

        if self.motion_key_body_indexes:
            ref_key_body_pos = (
                body_pos[:, self.motion_key_body_indexes]              # (N, 4, 3)
                - ref_root_pos.unsqueeze(1)
            ).reshape(num_samples, -1)                                 # (N, 12)
        else:
            ref_key_body_pos = torch.zeros(num_samples, 12, device=self.device)

        return torch.cat([
            ref_joint_pos,      # 12
            ref_joint_vel,      # 12
            base_height,        #  1
            projected_gravity,  #  3
            ref_key_body_pos,   # 12
        ], dim=-1)  # 40 — matches amp_observation_space

    def _reset_idx(self, env_ids: torch.Tensor):
        # Guard: Isaac Lab may pass None for a full-env reset
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # robot.reset() clears internal data buffers first, then super() resets
        # episode counters — matches robot_lab convention
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "random-start":
            # Reset to a random time in the motion to improve exploration
            times = np.random.uniform(0, self._motion_loader.duration, len(env_ids))
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
