# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP environment.

Rewards mirror robot_lab Go2 flat env:
  track_lin_vel_xy, track_ang_vel_z, upward, lin_vel_z_l2, ang_vel_xy_l2,
  feet_air_time, feet_air_time_variance, feet_gait, feet_slide,
  feet_contact_no_cmd, action_rate_l2, joint_torques_l2, joint_acc_l2,
  joint_pos_limits, stand_still, undesired_contacts, contact_forces,
  termination + imitation (joint_pos, joint_vel).

Obs:
  Actor  (41D): joint_pos+vel + base_height + proj_gravity + key_body_pos + progress
  Critic (73D): actor + base_lin_vel + feet_contact + height_scan(5x5)
  AMP    (40D): actor minus progress (all realizable on real robot)
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
        progress    = (self.episode_length_buf.float() / (self.max_episode_length - 1)).unsqueeze(-1)
        root_pos_w  = self.robot.data.body_pos_w[:, self.ref_body_index]   # (N, 3)
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]  # (N, 4) wxyz

        # base height (env-relative z)
        base_height = (root_pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1)  # (N,1)

        # projected gravity → body frame (IMU-measurable)
        gravity_w       = torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)
        proj_gravity    = _quat_rotate_inverse(root_quat_w, gravity_w)    # (N,3)

        # key body positions relative to root (FK-computable)
        key_body_rel = (
            self.robot.data.body_pos_w[:, self.key_body_indexes]
            - root_pos_w.unsqueeze(1)
        ).reshape(self.num_envs, -1)                                       # (N,12)

        # ── AMP obs (40D) ─────────────────────────────────────────────────────
        amp_obs = torch.cat([
            self.robot.data.joint_pos,   # 12
            self.robot.data.joint_vel,   # 12
            base_height,                 #  1
            proj_gravity,                #  3
            key_body_rel,                # 12
        ], dim=-1)  # 40

        # ── Actor obs (44D) ───────────────────────────────────────────────────
        policy_obs = torch.cat([amp_obs, self.commands, progress], dim=-1)  # 44

        # ── Critic privileged obs (73D) ───────────────────────────────────────
        base_lin_vel = _quat_rotate_inverse(
            root_quat_w, self.robot.data.root_lin_vel_w
        )  # (N,3) body frame

        # feet contact forces: net force magnitude per foot
        feet_contact = torch.zeros(self.num_envs, 4, device=self.device)
        if hasattr(self.contact_sensor.data, "net_forces_w"):
            # body indexes in contact sensor may differ from articulation — use body name match
            for i, foot_idx in enumerate(self.foot_body_indexes):
                try:
                    cf_idx = self.contact_sensor.data.body_names.index(
                        self.robot.data.body_names[foot_idx]
                    )
                    feet_contact[:, i] = self.contact_sensor.data.net_forces_w[:, cf_idx].norm(dim=-1)
                except (ValueError, AttributeError):
                    pass

        height_scan = self._get_height_scan(root_pos_w)  # (N,25)

        critic_obs = torch.cat([policy_obs, base_lin_vel, feet_contact, height_scan], dim=-1)  # 73

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

    def _get_height_scan(self, root_pos_w: torch.Tensor) -> torch.Tensor:
        """5×5 height scan, 0.25m spacing. Returns height relative to robot base."""
        offsets = torch.linspace(-0.5, 0.5, 5, device=self.device)
        gx, gy  = torch.meshgrid(offsets, offsets, indexing="ij")
        # flat ground: ground_z = env_origin_z
        ground_z  = self.scene.env_origins[:, 2]                    # (N,)
        robot_z   = root_pos_w[:, 2]                                # (N,)
        # height of ground relative to robot base (negative = ground is below)
        h_rel = (ground_z - robot_z).unsqueeze(-1).expand(-1, 25)   # (N,25)
        return h_rel

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg

        root_pos_w  = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]

        base_lin_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        base_ang_vel_b = _quat_rotate_inverse(root_quat_w, self.robot.data.root_ang_vel_w)

        cmd_vx  = self.commands[:, 0]
        cmd_vy  = self.commands[:, 1]
        cmd_wz  = self.commands[:, 2]
        cmd_zero = (self.commands.abs().sum(dim=-1) < 0.1)  # near-zero command

        # 1. Velocity tracking
        lin_err  = (cmd_vx - base_lin_vel_b[:, 0]).pow(2) + (cmd_vy - base_lin_vel_b[:, 1]).pow(2)
        ang_err  = (cmd_wz - base_ang_vel_b[:, 2]).pow(2)
        rew_track_lin = cfg.rew_track_lin_vel_xy * torch.exp(-lin_err / cfg.track_vel_sigma)
        rew_track_ang = cfg.rew_track_ang_vel_z  * torch.exp(-ang_err / cfg.track_vel_sigma)

        # 2. Upward (keep base upright)
        proj_gravity_b = _quat_rotate_inverse(
            root_quat_w,
            torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, 3)
        )
        rew_upward = cfg.rew_upward * torch.clamp(-proj_gravity_b[:, 2], 0.0, 1.0)

        # 3. Penalise vertical base velocity and roll/pitch rates
        rew_lin_vel_z  = cfg.rew_lin_vel_z_l2  * self.robot.data.root_lin_vel_w[:, 2].pow(2)
        rew_ang_vel_xy = cfg.rew_ang_vel_xy_l2 * base_ang_vel_b[:, :2].pow(2).sum(dim=-1)

        # 3b. Base height tracking — prevents body going too high or too low
        base_height = root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        rew_base_height = cfg.rew_base_height_l2 * (base_height - cfg.base_height_target).pow(2)

        # 3c. Flat orientation — penalise body tilt (roll/pitch via projected gravity)
        # proj_gravity_b[2] = -1 when perfectly upright, deviation from -1 = tilt
        rew_flat_orient = cfg.rew_flat_orientation_l2 * (proj_gravity_b[:, :2].pow(2).sum(dim=-1))

        # Resolve contact sensor indexes (needed by multiple rewards below)
        self._resolve_contact_indexes()
        cf_idx = self._foot_cf_indexes

        # 3d. Feet swing height — penalise deviation from target (robot_lab style)
        # Only penalise when foot is moving horizontally (swing phase)
        feet_pos_z = self.robot.data.body_pos_w[:, self.foot_body_indexes, 2]  # (N, 4)
        ground_z   = self.scene.env_origins[:, 2:3]  # (N, 1)
        foot_z_err = (feet_pos_z - ground_z - cfg.feet_height_target).pow(2)  # (N, 4)
        foot_xy_vel = self.robot.data.body_lin_vel_w[:, self.foot_body_indexes, :2]  # (N, 4, 2)
        foot_speed  = foot_xy_vel.norm(dim=-1)  # (N, 4)
        vel_weight  = torch.tanh(cfg.feet_height_tanh_mult * foot_speed)  # 0~1, active when moving
        rew_feet_height = cfg.rew_feet_height * (foot_z_err * vel_weight).sum(dim=-1)
        # Zero reward when command ≈ 0
        rew_feet_height *= (~cmd_zero).float()

        # 4. Feet air time (reward symmetric swing)
        air_time = torch.zeros(self.num_envs, 4, device=self.device)
        if cf_idx and self.contact_sensor.data.current_air_time is not None:
            try:
                air_time = self.contact_sensor.data.current_air_time[:, cf_idx]
            except IndexError:
                pass
        at_threshold = (air_time - cfg.feet_air_time_threshold).clamp(min=0.0)
        rew_air_time = cfg.rew_feet_air_time * at_threshold.sum(dim=-1) * (~cmd_zero).float()

        # 5. Feet air time variance (penalise asymmetry)
        rew_air_var = cfg.rew_feet_air_time_variance * air_time.var(dim=-1)

        # 6. Feet gait (diagonal pairs: FL+RR, FR+RL in sync)
        rew_gait = torch.zeros(self.num_envs, device=self.device)
        contact_bool = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        if cf_idx and hasattr(self.contact_sensor.data, "net_forces_w"):
            contact_bool = self.contact_sensor.data.net_forces_w[:, cf_idx, 2] > 1.0
            # FL=0, FR=1, RL=2, RR=3
            gait_sync = (contact_bool[:, 0].float() * contact_bool[:, 3].float()
                       + contact_bool[:, 1].float() * contact_bool[:, 2].float()) * 0.5
            rew_gait = cfg.rew_feet_gait * gait_sync

        # 7. Feet slide (penalise foot velocity when in contact)
        rew_slide = torch.zeros(self.num_envs, device=self.device)
        if cf_idx:
            foot_vel   = self.robot.data.body_lin_vel_w[:, self.foot_body_indexes, :2]  # (N,4,2)
            foot_speed = foot_vel.norm(dim=-1)                                           # (N,4)
            in_contact = contact_bool.float()
            rew_slide  = cfg.rew_feet_slide * (foot_speed * in_contact).sum(dim=-1)

        # 8. Stand still when cmd≈0
        rew_stand = cfg.rew_stand_still * cmd_zero.float() * \
            self.robot.data.joint_vel.pow(2).sum(dim=-1)

        # 9. Contact without command (encourage standing when cmd=0)
        rew_no_cmd = cfg.rew_feet_contact_no_cmd * cmd_zero.float()

        # 10. Undesired contacts (base should not touch ground)
        rew_undesired = torch.zeros(self.num_envs, device=self.device)
        if self._base_cf_index is not None and hasattr(self.contact_sensor.data, "net_forces_w"):
            base_contact = (
                self.contact_sensor.data.net_forces_w[:, self._base_cf_index].norm(dim=-1) > 1.0
            ).float()
            rew_undesired = cfg.rew_undesired_contacts * base_contact

        # 11. Excessive contact forces on feet
        rew_cf = torch.zeros(self.num_envs, device=self.device)
        if cf_idx and hasattr(self.contact_sensor.data, "net_forces_w"):
            foot_forces = self.contact_sensor.data.net_forces_w[:, cf_idx].norm(dim=-1)  # (N,4)
            rew_cf = cfg.rew_contact_forces * foot_forces.clamp(min=100.0).sum(dim=-1)

        # 12. Action rate (smoothness)
        rew_action_rate = cfg.rew_action_rate_l2 * (self.actions - self.last_actions).pow(2).sum(dim=-1)

        # 13. Joint torques
        rew_torques = cfg.rew_joint_torques_l2 * self.robot.data.applied_torque.pow(2).sum(dim=-1)

        # 14. Joint accelerations
        joint_acc = (self.robot.data.joint_vel - self.last_joint_vel) / (self.cfg.decimation * self.cfg.dt)
        rew_joint_acc = cfg.rew_joint_acc_l2 * joint_acc.pow(2).sum(dim=-1)

        # 15. Joint position limits — continuous penalty (distance beyond soft limits)
        jpos   = self.robot.data.joint_pos
        lo     = self.robot.data.soft_joint_pos_limits[0, :, 0]
        hi     = self.robot.data.soft_joint_pos_limits[0, :, 1]
        below  = (lo - jpos).clamp(min=0.0)  # how far below lower limit
        above  = (jpos - hi).clamp(min=0.0)  # how far above upper limit
        out    = (below.pow(2) + above.pow(2)).sum(dim=-1)  # sum of squared violations
        rew_limits = cfg.rew_joint_pos_limits * out

        # 16. Imitation (keeps style reward grounded in reference motion)
        with torch.no_grad():
            times = (self.episode_length_buf * self.step_dt).cpu().numpy()
            ref_dof_pos, ref_dof_vel, _, _, _, _ = self._motion_loader.sample(
                num_samples=self.num_envs, times=times
            )
        ref_jp = ref_dof_pos[:, self.motion_dof_indexes]
        ref_jv = ref_dof_vel[:, self.motion_dof_indexes]
        rew_imit_jp = cfg.rew_imitation_joint_pos * torch.exp(
            -jpos.sub(ref_jp).pow(2).sum(dim=-1) / cfg.imitation_sigma_joint_pos
        )
        rew_imit_jv = cfg.rew_imitation_joint_vel * torch.exp(
            -self.robot.data.joint_vel.sub(ref_jv).pow(2).sum(dim=-1) / cfg.imitation_sigma_joint_vel
        )

        # 17. Termination penalty
        base_height = root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        rew_term = cfg.rew_termination * (base_height < cfg.termination_height).float()

        # Cache for next step
        self.last_actions.copy_(self.actions)
        self.last_joint_vel.copy_(self.robot.data.joint_vel)

        # Multiply by step_dt to match robot_lab RewardManager: value = weight * dt * func()
        # This keeps value function estimates at the same scale as robot_lab
        dt = self.step_dt
        total = dt * (
            rew_track_lin + rew_track_ang
            + rew_upward + rew_lin_vel_z + rew_ang_vel_xy
            + rew_base_height + rew_flat_orient + rew_feet_height
            + rew_air_time + rew_air_var + rew_gait + rew_slide
            + rew_stand + rew_no_cmd
            + rew_undesired + rew_cf
            + rew_action_rate + rew_torques + rew_joint_acc + rew_limits
            + rew_imit_jp + rew_imit_jv
        ) + rew_term

        # Per-term breakdown for logging (per-step mean, matching robot_lab Episode_Reward format)
        self.extras["reward_terms"] = {
            "track_lin_vel":  rew_track_lin.mean().item(),
            "track_ang_vel":  rew_track_ang.mean().item(),
            "upward":         rew_upward.mean().item(),
            "lin_vel_z":      rew_lin_vel_z.mean().item(),
            "ang_vel_xy":     rew_ang_vel_xy.mean().item(),
            "base_height":    rew_base_height.mean().item(),
            "flat_orient":    rew_flat_orient.mean().item(),
            "feet_swing_h":   rew_feet_height.mean().item(),
            "feet_air_time":  rew_air_time.mean().item(),
            "feet_air_var":   rew_air_var.mean().item(),
            "feet_gait":      rew_gait.mean().item(),
            "feet_slide":     rew_slide.mean().item(),
            "stand_still":    rew_stand.mean().item(),
            "action_rate":    rew_action_rate.mean().item(),
            "joint_torques":  rew_torques.mean().item(),
            "joint_acc":      rew_joint_acc.mean().item(),
            "joint_limits":   rew_limits.mean().item(),
            "imitation_jp":   rew_imit_jp.mean().item(),
            "imitation_jv":   rew_imit_jv.mean().item(),
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
        """Sample AMP obs from reference motion (40D, matches env amp_obs)."""
        times = np.random.uniform(0, self._motion_loader.duration, num_samples)
        dof_pos, dof_vel, body_pos, body_rot, _, _ = self._motion_loader.sample(
            num_samples=num_samples, times=times
        )
        ref_jp   = dof_pos[:, self.motion_dof_indexes]
        ref_jv   = dof_vel[:, self.motion_dof_indexes]
        ref_rpos = body_pos[:, self.motion_ref_body_index]   # (N,3)
        ref_rquat= body_rot[:, self.motion_ref_body_index]   # (N,4)

        base_height = ref_rpos[:, 2:3]                        # (N,1)
        gravity_w   = torch.tensor([0., 0., -1.], device=self.device).expand(num_samples, 3)
        proj_grav   = _quat_rotate_inverse(ref_rquat, gravity_w)  # (N,3)

        if self.motion_key_body_indexes:
            ref_key = (
                body_pos[:, self.motion_key_body_indexes] - ref_rpos.unsqueeze(1)
            ).reshape(num_samples, -1)
        else:
            ref_key = torch.zeros(num_samples, 12, device=self.device)

        return torch.cat([ref_jp, ref_jv, base_height, proj_grav, ref_key], dim=-1)  # 40
