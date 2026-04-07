# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP environment.

Rewards mirror robot_lab Go2 flat env:
  track_lin_vel_xy, track_ang_vel_z, upward, lin_vel_z_l2, ang_vel_xy_l2,
  feet_air_time, feet_air_time_variance, feet_gait, feet_slide,
  feet_contact_no_cmd, action_rate_l2, joint_torques_l2, joint_acc_l2,
  joint_pos_limits, stand_still, undesired_contacts, contact_forces,
  termination + imitation (joint_pos, joint_vel).

Obs (aligned with robot_lab):
  Actor  (45D): base_ang_vel + proj_gravity + cmd + joint_pos_rel + joint_vel + last_actions
  Critic (48D): actor + base_lin_vel (privileged)
  AMP    (30D): joint_pos_rel + joint_vel + proj_gravity + base_ang_vel
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .orix_amp_env_cfg import OrixAmpEnvCfg

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from motions.motion_loader_quad import QuadMotionLoader


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q^{-1}. q is wxyz, v is (N,3). Returns (N,3) in body frame."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    xyz = torch.cat([x, y, z], dim=-1)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ── Environment ───────────────────────────────────────────────────────────────

class OrixAmpEnv(DirectRLEnv):
    cfg: OrixAmpEnvCfg

    def __init__(self, cfg: OrixAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action offset = default standing pose, per-joint scale (match robot_lab)
        self.action_offset = self.robot.data.default_joint_pos[0].clone()
        # hip: 0.1 rad, thigh/calf: 0.2 rad (robot_lab action_scale config)
        self.action_scale = torch.ones(cfg.action_space, device=self.device) * 0.2
        for i, name in enumerate(self.robot.data.joint_names):
            if "hip" in name:
                self.action_scale[i] = 0.1
        self.action_clip = 3.0  # match robot_lab ±3.0
        self.last_actions   = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.last_joint_vel = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # Velocity commands (sampled per reset)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)  # vx, vy, wz
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Motion loader
        self._motion_loader = QuadMotionLoader(
            motion_file=self.cfg.motion_file, device=self.device
        )

        # Body / joint indices
        self.ref_body_index    = self.robot.data.body_names.index(self.cfg.reference_body)
        self.foot_body_names   = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        self.foot_body_indexes = [self.robot.data.body_names.index(n) for n in self.foot_body_names]

        # Key bodies for AMP obs (same as feet)
        self.key_body_indexes = self.foot_body_indexes

        # Contact sensor body indexes — separate from robot body indexes
        # Must be resolved AFTER scene is set up and sensor data is available
        self._foot_cf_indexes: list[int] | None = None
        self._base_cf_index:   int       | None = None

        # Motion DOF mapping
        self.motion_dof_indexes      = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index   = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(
            [n for n in self.foot_body_names if n in self._motion_loader.body_names]
        )

        # AMP observation buffer (3 frames stacked)
        self.amp_observation_size   = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_buffer = torch.zeros(
            self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space,
            device=self.device,
        )

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

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
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── Action ────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-self.action_clip, self.action_clip)

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        cfg = self.cfg
        root_pos_w  = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]

        # Body-frame velocities
        base_ang_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        base_lin_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_lin_vel_w)

        # Projected gravity (IMU)
        gravity_w    = torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)
        proj_gravity = _quat_rotate_inverse(root_quat_w, gravity_w)

        # Joint pos/vel relative to default
        joint_pos_rel = self.robot.data.joint_pos - self.action_offset
        joint_vel     = self.robot.data.joint_vel

        # ── AMP obs (30D) — extractable from motion data ─────────────────────
        amp_obs = torch.cat([
            joint_pos_rel * cfg.obs_scale_joint_pos,   # 12
            joint_vel     * cfg.obs_scale_joint_vel,    # 12
            proj_gravity,                               #  3
            base_ang_vel_b * cfg.obs_scale_ang_vel,     #  3
        ], dim=-1)  # 30

        # ── Actor obs (45D) — match robot_lab ─────────────────────────────────
        policy_obs = torch.cat([
            base_ang_vel_b * cfg.obs_scale_ang_vel,     #  3
            proj_gravity,                               #  3
            self.commands,                              #  3
            joint_pos_rel * cfg.obs_scale_joint_pos,    # 12
            joint_vel     * cfg.obs_scale_joint_vel,    # 12
            self.last_actions,                          # 12
        ], dim=-1)  # 45

        # ── Critic obs (48D) — actor + privileged ─────────────────────────────
        critic_obs = torch.cat([
            policy_obs,        # 45
            base_lin_vel_b,    #  3  (privileged: ground truth from sim)
        ], dim=-1)  # 48

        # ── Update AMP history ────────────────────────────────────────────────
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs
        # Preserve reward_terms written by _get_rewards() (called before _get_observations in Isaac Lab)
        prev_reward_terms = self.extras.get("reward_terms", {}) if hasattr(self, "extras") else {}
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size),
            "reward_terms": prev_reward_terms,
        }

        return {"policy": policy_obs, "critic": critic_obs}

    def _resolve_contact_indexes(self):
        """Lazily resolve contact sensor body indexes using ContactSensor.find_bodies()."""
        if self._foot_cf_indexes is not None:
            return
        try:
            # body_names is on ContactSensor, not ContactSensorData
            foot_ids, _ = self.contact_sensor.find_bodies(self.foot_body_names)
            base_ids, _ = self.contact_sensor.find_bodies(["base_link"])
            self._foot_cf_indexes = foot_ids
            self._base_cf_index   = base_ids[0] if base_ids else None
            print(f"[OrixAmpEnv] contact sensor foot_cf_indexes={self._foot_cf_indexes} base={self._base_cf_index}")
        except (ValueError, AttributeError) as e:
            print(f"[OrixAmpEnv] contact sensor index resolution failed: {e}")
            self._foot_cf_indexes = []
            self._base_cf_index   = None

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """Reward function aligned with robot_lab orix_dog rough_env_cfg."""
        cfg = self.cfg

        root_pos_w  = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]

        base_lin_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        base_ang_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_ang_vel_w)

        jpos = self.robot.data.joint_pos
        jvel = self.robot.data.joint_vel
        cmd_vx, cmd_vy, cmd_wz = self.commands[:, 0], self.commands[:, 1], self.commands[:, 2]
        cmd_zero = (self.commands.abs().sum(dim=-1) < 0.1)

        # Contact sensor
        self._resolve_contact_indexes()
        cf_idx = self._foot_cf_indexes
        contact_bool = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        if cf_idx and hasattr(self.contact_sensor.data, "net_forces_w"):
            contact_bool = self.contact_sensor.data.net_forces_w[:, cf_idx, 2] > 1.0

        # ── Velocity tracking ─────────────────────────────────────────────────
        lin_err = (cmd_vx - base_lin_vel_b[:, 0]).pow(2) + (cmd_vy - base_lin_vel_b[:, 1]).pow(2)
        ang_err = (cmd_wz - base_ang_vel_b[:, 2]).pow(2)
        rew_track_lin = cfg.rew_track_lin_vel_xy * torch.exp(-lin_err / cfg.track_vel_sigma)
        rew_track_ang = cfg.rew_track_ang_vel_z  * torch.exp(-ang_err / cfg.track_vel_sigma)

        # ── Posture ───────────────────────────────────────────────────────────
        proj_gravity_b = _quat_rotate_inverse(
            root_quat_w, torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)
        )
        # robot_lab: (1 - proj_gravity_b[2])^2  — direct copy
        rew_upward     = cfg.rew_upward * (1.0 - proj_gravity_b[:, 2]).pow(2)
        rew_lin_vel_z  = cfg.rew_lin_vel_z_l2  * self.robot.data.root_lin_vel_w[:, 2].pow(2)
        rew_ang_vel_xy = cfg.rew_ang_vel_xy_l2 * base_ang_vel_b[:, :2].pow(2).sum(dim=-1)

        # gravity clamp factor used by several rewards (robot_lab pattern)
        grav_clamp = torch.clamp(-proj_gravity_b[:, 2], 0.0, 0.7) / 0.7

        # ── Feet height body (robot_lab style) ────────────────────────────────
        feet_pos_body = (
            self.robot.data.body_pos_w[:, self.foot_body_indexes] - root_pos_w.unsqueeze(1)
        )
        feet_pos_body_frame = torch.zeros_like(feet_pos_body)
        for i in range(4):
            feet_pos_body_frame[:, i] = _quat_rotate_inverse(root_quat_w, feet_pos_body[:, i])
        foot_z_err = (feet_pos_body_frame[:, :, 2] - cfg.feet_height_body_target).pow(2)
        foot_xy_vel = self.robot.data.body_lin_vel_w[:, self.foot_body_indexes, :2]
        vel_weight = torch.tanh(cfg.feet_height_body_tanh_mult * foot_xy_vel.norm(dim=-1))
        rew_feet_hb = cfg.rew_feet_height_body * (foot_z_err * vel_weight).sum(dim=-1)
        rew_feet_hb *= (~cmd_zero).float()

        # ── Feet air time (robot_lab: only on first contact, uses last_air_time) ──
        rew_air_time = torch.zeros(self.num_envs, device=self.device)
        rew_air_var  = torch.zeros(self.num_envs, device=self.device)
        if cf_idx:
            first_contact = self.contact_sensor.compute_first_contact(self.step_dt)[:, cf_idx]
            last_air = self.contact_sensor.data.last_air_time[:, cf_idx]
            last_contact = self.contact_sensor.data.last_contact_time[:, cf_idx]
            cmd_norm = self.commands.norm(dim=-1) > 0.1
            rew_air_time = cfg.rew_feet_air_time * (
                ((last_air - cfg.feet_air_time_threshold) * first_contact.float()).sum(dim=-1)
                * cmd_norm.float() * grav_clamp
            )
            # variance penalty: var(last_air) + var(last_contact), clamped at 0.5
            air_clipped = last_air.clamp(max=0.5)
            con_clipped = last_contact.clamp(max=0.5)
            rew_air_var = cfg.rew_feet_air_time_variance * (
                (air_clipped.var(dim=-1) + con_clipped.var(dim=-1)) * grav_clamp
            )

        # ── Feet gait (robot_lab GaitReward — sync diagonal pairs, async cross) ──
        # Synced pairs: (FL, RR) and (FR, RL). Indices: FL=0, FR=1, RL=2, RR=3
        rew_gait = torch.zeros(self.num_envs, device=self.device)
        if cf_idx and self.contact_sensor.data.current_air_time is not None:
            air = self.contact_sensor.data.current_air_time[:, cf_idx]
            con = self.contact_sensor.data.current_contact_time[:, cf_idx]
            max_sq = cfg.feet_gait_max_err ** 2

            def sync_r(a, b):
                se_air = (air[:, a] - air[:, b]).pow(2).clamp(max=max_sq)
                se_con = (con[:, a] - con[:, b]).pow(2).clamp(max=max_sq)
                return torch.exp(-(se_air + se_con) / cfg.feet_gait_std)

            def async_r(a, b):
                se_0 = (air[:, a] - con[:, b]).pow(2).clamp(max=max_sq)
                se_1 = (con[:, a] - air[:, b]).pow(2).clamp(max=max_sq)
                return torch.exp(-(se_0 + se_1) / cfg.feet_gait_std)

            # Synced: FL-RR (0,3) and FR-RL (1,2)
            sync = sync_r(0, 3) * sync_r(1, 2)
            # Async: 4 cross pairs
            async_r0 = async_r(0, 1)  # FL vs FR
            async_r1 = async_r(3, 2)  # RR vs RL
            async_r2 = async_r(0, 2)  # FL vs RL
            async_r3 = async_r(1, 3)  # FR vs RR
            asyn = async_r0 * async_r1 * async_r2 * async_r3

            cmd_active = (self.commands.norm(dim=-1) > 0.06) | \
                         (self.robot.data.root_lin_vel_w[:, :2].norm(dim=-1) > 0.5)
            rew_gait = cfg.rew_feet_gait * torch.where(
                cmd_active, sync * asyn, torch.zeros_like(sync)
            ) * grav_clamp

        # ── Feet slide (robot_lab: body-frame foot lateral velocity × contact) ───
        rew_slide = torch.zeros(self.num_envs, device=self.device)
        if cf_idx:
            # Foot velocity relative to root, transformed to body frame
            foot_vel_w = self.robot.data.body_lin_vel_w[:, self.foot_body_indexes]      # (N,4,3)
            root_lin_vel_w = self.robot.data.root_lin_vel_w.unsqueeze(1)                # (N,1,3)
            foot_vel_rel_w = foot_vel_w - root_lin_vel_w                                # (N,4,3)
            foot_vel_b = torch.stack(
                [_quat_rotate_inverse(root_quat_w, foot_vel_rel_w[:, i]) for i in range(4)],
                dim=1
            )                                                                            # (N,4,3)
            foot_lateral = foot_vel_b[:, :, :2].norm(dim=-1)                             # (N,4)
            rew_slide = cfg.rew_feet_slide * (foot_lateral * contact_bool.float()).sum(dim=-1)

        # ── Stand still (robot_lab: joint_deviation_l1 × cmd_small × grav_clamp) ──
        joint_dev_l1 = (jpos - self.action_offset).abs().sum(dim=-1)
        cmd_small = (self.commands.norm(dim=-1) < 0.06).float()
        rew_stand = cfg.rew_stand_still * joint_dev_l1 * cmd_small * grav_clamp
        # robot_lab: sum(first_contact) when cmd<0.1 × grav_clamp
        rew_no_cmd = torch.zeros(self.num_envs, device=self.device)
        if cf_idx:
            first_contact_no_cmd = self.contact_sensor.compute_first_contact(self.step_dt)[:, cf_idx]
            cmd_zero_strict = (self.commands.norm(dim=-1) < 0.1).float()
            rew_no_cmd = cfg.rew_feet_contact_no_cmd * \
                first_contact_no_cmd.float().sum(dim=-1) * cmd_zero_strict * grav_clamp

        # ── Contacts ──────────────────────────────────────────────────────────
        rew_undesired = torch.zeros(self.num_envs, device=self.device)
        if self._base_cf_index is not None and hasattr(self.contact_sensor.data, "net_forces_w"):
            base_f = self.contact_sensor.data.net_forces_w[:, self._base_cf_index].norm(dim=-1)
            rew_undesired = cfg.rew_undesired_contacts * (base_f > 1.0).float()
        rew_cf = torch.zeros(self.num_envs, device=self.device)
        if cf_idx and hasattr(self.contact_sensor.data, "net_forces_w"):
            foot_f = self.contact_sensor.data.net_forces_w[:, cf_idx].norm(dim=-1)
            rew_cf = cfg.rew_contact_forces * foot_f.clamp(min=100.0).sum(dim=-1)

        # ── Regularisation ────────────────────────────────────────────────────
        rew_action_rate = cfg.rew_action_rate_l2 * (self.actions - self.last_actions).pow(2).sum(dim=-1)
        rew_torques     = cfg.rew_joint_torques_l2 * self.robot.data.applied_torque.pow(2).sum(dim=-1)
        joint_acc       = (jvel - self.last_joint_vel) / (cfg.decimation * cfg.dt)
        rew_joint_acc   = cfg.rew_joint_acc_l2 * joint_acc.pow(2).sum(dim=-1)

        # Joint power = torque × velocity
        rew_power = cfg.rew_joint_power * (self.robot.data.applied_torque * jvel).abs().sum(dim=-1)

        # Joint position penalty (robot_lab: norm of deviation, scale up when standing still)
        # robot_lab uses norm (L2), not squared sum, and scales by stand_still_scale when stationary
        running_pen = (jpos - self.action_offset).norm(dim=-1)
        body_vel = self.robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
        is_running = (self.commands.norm(dim=-1) > 0.06) | (body_vel > 0.5)
        # robot_lab uses stand_still_scale=5.0 to amplify when stationary
        scale = torch.where(is_running, torch.ones_like(running_pen), 5.0 * torch.ones_like(running_pen))
        rew_pos_pen = cfg.rew_joint_pos_penalty * scale * running_pen

        # Joint mirror (robot_lab: (joint_a + joint_b)^2, sum=0 when mirrored)
        # robot_lab pairs: [FR, RL] and [FL, RR]
        # Joint order in robot: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
        mirror_err = (jpos[:, 3:6] + jpos[:, 6:9]).pow(2).sum(dim=-1) + \
                     (jpos[:, 0:3] + jpos[:, 9:12]).pow(2).sum(dim=-1)
        rew_mirror = cfg.rew_joint_mirror * mirror_err

        # Joint limits (continuous)
        lo = self.robot.data.soft_joint_pos_limits[0, :, 0]
        hi = self.robot.data.soft_joint_pos_limits[0, :, 1]
        below = (lo - jpos).clamp(min=0.0)
        above = (jpos - hi).clamp(min=0.0)
        rew_limits = cfg.rew_joint_pos_limits * (below.pow(2) + above.pow(2)).sum(dim=-1)

        # ── Termination penalty ───────────────────────────────────────────────
        base_height = root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        rew_term = cfg.rew_termination * (base_height < cfg.termination_height).float()

        # Cache
        self.last_actions.copy_(self.actions)
        self.last_joint_vel.copy_(jvel)

        # ── Total (dt-scaled, match robot_lab RewardManager) ──────────────────
        dt = self.step_dt
        total = dt * (
            rew_track_lin + rew_track_ang
            + rew_upward + rew_lin_vel_z + rew_ang_vel_xy
            + rew_feet_hb + rew_air_time + rew_air_var + rew_gait + rew_slide
            + rew_stand + rew_no_cmd + rew_undesired + rew_cf
            + rew_action_rate + rew_torques + rew_joint_acc
            + rew_power + rew_pos_pen + rew_mirror + rew_limits
        ) + rew_term

        self.extras["reward_terms"] = {
            "track_lin_vel":  rew_track_lin.mean().item(),
            "track_ang_vel":  rew_track_ang.mean().item(),
            "upward":         rew_upward.mean().item(),
            "lin_vel_z":      rew_lin_vel_z.mean().item(),
            "ang_vel_xy":     rew_ang_vel_xy.mean().item(),
            "feet_height_b":  rew_feet_hb.mean().item(),
            "feet_air_time":  rew_air_time.mean().item(),
            "feet_air_var":   rew_air_var.mean().item(),
            "feet_gait":      rew_gait.mean().item(),
            "feet_slide":     rew_slide.mean().item(),
            "stand_still":    rew_stand.mean().item(),
            "action_rate":    rew_action_rate.mean().item(),
            "joint_torques":  rew_torques.mean().item(),
            "joint_acc":      rew_joint_acc.mean().item(),
            "joint_power":    rew_power.mean().item(),
            "joint_pos_pen":  rew_pos_pen.mean().item(),
            "joint_mirror":   rew_mirror.mean().item(),
            "joint_limits":   rew_limits.mean().item(),
            "termination":    rew_term.mean().item(),
        }
        return total

    # ── Termination ───────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out   = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.early_termination:
            root_z     = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
            base_height = root_z - self.scene.env_origins[:, 2]
            terminated = base_height < self.cfg.termination_height
        return terminated, time_out

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # Random initial pose from motion reference
        if self.cfg.reset_strategy == "random-start":
            times = np.random.uniform(0, self._motion_loader.duration, n)
            dof_pos, dof_vel, body_pos, body_rot, _, _ = self._motion_loader.sample(
                num_samples=n, times=times
            )
            env_origins = self.scene.env_origins[env_ids]
            root_pos    = body_pos[:, self.motion_ref_body_index] + env_origins
            root_quat   = body_rot[:, self.motion_ref_body_index]
            ref_jp      = dof_pos[:, self.motion_dof_indexes]
            ref_jv      = dof_vel[:, self.motion_dof_indexes]
            self.robot.write_root_pose_to_sim(
                torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
            )
            self.robot.write_joint_state_to_sim(ref_jp, ref_jv, env_ids=env_ids)
        else:
            default_root = self.robot.data.default_root_state[env_ids].clone()
            default_root[:, :3] += self.scene.env_origins[env_ids]
            self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
            self.robot.write_joint_state_to_sim(
                self.robot.data.default_joint_pos[env_ids],
                self.robot.data.default_joint_vel[env_ids],
                env_ids=env_ids,
            )

        self._resample_commands(env_ids)
        self.last_actions[env_ids]   = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.amp_observation_buffer[env_ids] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor):
        n = len(env_ids)
        cfg = self.cfg
        self.commands[env_ids, 0] = torch.FloatTensor(n).uniform_(*cfg.cmd_lin_vel_x_range).to(self.device)
        self.commands[env_ids, 1] = torch.FloatTensor(n).uniform_(*cfg.cmd_lin_vel_y_range).to(self.device)
        self.commands[env_ids, 2] = torch.FloatTensor(n).uniform_(*cfg.cmd_ang_vel_z_range).to(self.device)

    # ── AMP reference sampling ────────────────────────────────────────────────

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Sample AMP obs from reference motion (30D, matches env amp_obs).

        Format: joint_pos_rel(12) + joint_vel(12) + proj_gravity(3) + base_ang_vel(3)
        """
        cfg = self.cfg
        times = np.random.uniform(0, self._motion_loader.duration, num_samples)
        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = \
            self._motion_loader.sample(num_samples=num_samples, times=times)

        ref_jp  = dof_pos[:, self.motion_dof_indexes]
        ref_jv  = dof_vel[:, self.motion_dof_indexes]
        ref_rquat = body_rot[:, self.motion_ref_body_index]  # (N, 4)

        # joint_pos relative to default
        ref_jp_rel = ref_jp - self.action_offset

        # projected gravity from reference quaternion
        gravity_w = torch.tensor([0., 0., -1.], device=self.device).expand(num_samples, 3)
        proj_grav = _quat_rotate_inverse(ref_rquat, gravity_w)

        # base angular velocity in body frame
        ref_ang_vel_w = body_ang_vel[:, self.motion_ref_body_index]
        ref_ang_vel_b = _quat_rotate_inverse(ref_rquat, ref_ang_vel_w)

        return torch.cat([
            ref_jp_rel * cfg.obs_scale_joint_pos,   # 12
            ref_jv     * cfg.obs_scale_joint_vel,    # 12
            proj_grav,                               #  3
            ref_ang_vel_b * cfg.obs_scale_ang_vel,   #  3
        ], dim=-1)  # 30
