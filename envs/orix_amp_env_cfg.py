# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP environment config.

Obs spaces:
  Actor  (41D): joint_pos(12)+joint_vel(12)+base_height(1)+proj_gravity(3)+key_body_pos(12)+progress(1)
  Critic (73D): actor(41)+base_lin_vel(3)+feet_contact(4)+height_scan(25)
  AMP    (40D): same as actor minus progress — all realizable on real robot
"""
from __future__ import annotations
import os

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions")

ORIX_DOG_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        asset_path=os.path.join(os.path.dirname(__file__), "..", "urdf", "orix_dog.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "FR_hip_joint": 0.0, "FR_thigh_joint": -0.3, "FR_calf_joint":  1.1,
            "FL_hip_joint": 0.0, "FL_thigh_joint":  0.3, "FL_calf_joint": -1.1,
            "RR_hip_joint": 0.0, "RR_thigh_joint": -0.3, "RR_calf_joint":  1.1,
            "RL_hip_joint": 0.0, "RL_thigh_joint":  0.3, "RL_calf_joint": -1.1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=23.7,     # match robot_lab
            saturation_effort=23.7,
            velocity_limit=30.0,   # match robot_lab
            stiffness=20.0,        # was 25 → match robot_lab
            damping=1.0,           # was 0.5 → match robot_lab (critical for damping oscillation)
            friction=0.0,
        ),
    },
)


@configclass
class OrixAmpEnvCfg(DirectRLEnvCfg):
    """Orix Dog AMP environment config."""

    # ── Obs / action spaces (aligned with robot_lab) ────────────────────────
    # actor (45D): base_ang_vel(3) + proj_gravity(3) + cmd(3) + joint_pos(12) + joint_vel(12) + last_actions(12)
    # critic (48D): actor(45) + base_lin_vel(3)
    # AMP (30D): joint_pos(12) + joint_vel(12) + proj_gravity(3) + base_ang_vel(3)
    observation_space = 45   # actor
    action_space      = 12
    state_space       = 48   # critic = actor + privileged
    num_amp_observations  = 1
    amp_observation_space = 30

    # Observation scales (match robot_lab)
    obs_scale_ang_vel:   float = 0.25
    obs_scale_joint_pos: float = 1.0
    obs_scale_joint_vel: float = 0.05

    # ── Env timing ────────────────────────────────────────────────────────────
    episode_length_s = 20.0    # 20s × 50Hz = 1000 steps (matches robot_lab)
    decimation       = 4       # policy step = 4 × 0.005s = 0.02s = 50Hz
    dt               = 0.005  # 5ms physics dt (matches robot_lab sim.dt)

    # ── Termination ───────────────────────────────────────────────────────────
    early_termination  = False  # disabled — let upward/base_height rewards handle posture
    termination_height = 0.15

    # ── Velocity command ranges (conservative for small robot, learn to walk first) ──
    cmd_lin_vel_x_range: tuple = (-0.5, 0.5)   # m/s
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)   # rad/s

    # ── Motion reference ──────────────────────────────────────────────────────
    motion_file    = os.path.join(MOTIONS_DIR, "orix_trot_medium_30.npz")
    reference_body = "base_link"
    reset_strategy = "random-start"

    # ── Reward weights (aligned with robot_lab orix_dog) ─────────────────────
    # Velocity tracking
    rew_track_lin_vel_xy: float = 6.0
    rew_track_ang_vel_z:  float = 3.0
    track_vel_sigma:      float = 0.25

    # Posture / stability
    rew_upward:              float =  1.0
    rew_lin_vel_z_l2:        float = -2.0
    rew_ang_vel_xy_l2:       float = -0.05

    # Foot behaviour (match robot_lab)
    rew_feet_air_time:          float =  0.3
    feet_air_time_threshold:    float =  0.1
    rew_feet_air_time_variance: float = -1.0
    rew_feet_gait:              float =  0.5
    rew_feet_slide:             float = -0.1
    rew_feet_contact_no_cmd:    float =  0.1
    rew_feet_height_body:       float = -1.0     # penalise feet too high relative to body
    feet_height_body_target:    float = -0.2
    feet_height_body_tanh_mult: float =  5.0

    # Regularisation (match robot_lab)
    rew_action_rate_l2:    float = -0.01
    rew_joint_torques_l2:  float = -2.5e-5
    rew_joint_acc_l2:      float = -2.5e-7
    rew_joint_pos_limits:  float = -5.0
    rew_joint_power:       float = -2e-5
    rew_joint_pos_penalty: float = -1.0
    rew_joint_mirror:      float = -0.05
    rew_stand_still:       float = -2.0

    # Contact penalties
    rew_undesired_contacts: float = -1.0
    rew_contact_forces:     float = -1.5e-4

    # Imitation (for AMP mode, off in pure PPO)
    rew_imitation_joint_pos: float = 0.0
    rew_imitation_joint_vel: float = 0.0
    imitation_sigma_joint_pos: float = 1.5
    imitation_sigma_joint_vel: float = 8.0

    # Termination penalty
    rew_termination: float = -10.0

    # ── Simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = ORIX_DOG_CFG

    # ── Contact sensor (all bodies, for feet contact detection) ───────────────
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
