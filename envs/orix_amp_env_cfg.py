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
        "hip_thigh": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=18.0,
            saturation_effort=18.0,
            velocity_limit=35.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=36.0,     # URDF: calf effort=36 (was wrongly 18)
            saturation_effort=36.0,
            velocity_limit=17.5,   # URDF: calf velocity=17.5 (was wrongly 35)
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


@configclass
class OrixAmpEnvCfg(DirectRLEnvCfg):
    """Orix Dog AMP environment config."""

    # ── Obs / action spaces ───────────────────────────────────────────────────
    # actor: joint_pos(12)+joint_vel(12)+height(1)+proj_grav(3)+key_body(12)+cmd(3)+progress(1) = 44
    # critic: actor(44)+base_lin_vel(3)+feet_contact(4)+height_scan(25) = 76
    # AMP: single frame, no cmd/progress = 40
    observation_space = 44
    action_space      = 12
    state_space       = 76
    num_amp_observations  = 1   # single frame — prevents discriminator exploiting temporal artifacts
    amp_observation_space = 40

    # ── Env timing ────────────────────────────────────────────────────────────
    episode_length_s = 20.0    # 20s × 50Hz = 1000 steps (matches robot_lab)
    decimation       = 4       # policy step = 4 × 0.005s = 0.02s = 50Hz
    dt               = 0.005  # 5ms physics dt (matches robot_lab sim.dt)

    # ── Termination ───────────────────────────────────────────────────────────
    early_termination  = True
    termination_height = 0.15  # base z below this → episode end

    # ── Velocity command ranges (conservative for small robot, learn to walk first) ──
    cmd_lin_vel_x_range: tuple = (-0.5, 0.5)   # m/s
    cmd_lin_vel_y_range: tuple = (-0.3, 0.3)
    cmd_ang_vel_z_range: tuple = (-0.5, 0.5)   # rad/s

    # ── Motion reference ──────────────────────────────────────────────────────
    motion_file    = os.path.join(MOTIONS_DIR, "orix_trot_medium_30.npz")
    reference_body = "base_link"
    reset_strategy = "random-start"

    # ── Reward weights ────────────────────────────────────────────────────────
    # Velocity tracking (task)
    rew_track_lin_vel_xy: float = 6.0
    rew_track_ang_vel_z:  float = 3.0
    track_vel_sigma:      float = 0.25

    # Posture / stability
    rew_upward:              float =  1.0
    rew_lin_vel_z_l2:        float = -2.0     # penalise vertical base velocity
    rew_ang_vel_xy_l2:       float = -0.05    # penalise roll/pitch rate
    rew_base_height_l2:      float = -5.0     # penalise deviation from target height
    base_height_target:      float =  0.21    # FK: thigh=0.3, calf=1.1 → standing 0.21m
    rew_flat_orientation_l2: float = -2.0     # penalise body tilt (roll/pitch)
    rew_feet_height_body:    float = -1.0     # penalise feet too high relative to body
    feet_height_body_target: float = -0.2     # feet should be ~0.2m below body

    # Foot behaviour — gait phase is critical
    rew_feet_air_time:          float =  0.5
    feet_air_time_threshold:    float =  0.3   # shorter for small robot
    rew_feet_air_time_variance: float = -1.5
    rew_feet_gait:              float =  1.0   # diagonal sync emphasis
    rew_feet_slide:             float = -0.15
    rew_feet_contact_no_cmd:    float =  0.1

    # Regularisation
    rew_action_rate_l2:   float = -0.01
    rew_joint_torques_l2: float = -2.5e-5
    rew_joint_acc_l2:     float = -2.5e-7
    rew_joint_pos_limits: float = -2.0
    rew_stand_still:      float = -0.5

    # Contact penalties
    rew_undesired_contacts: float = -1.0
    rew_contact_forces:     float = -1.5e-4

    # Imitation (AMP task reward component)
    rew_imitation_joint_pos: float = 1.0   # scaled down — velocity tracking is primary
    rew_imitation_joint_vel: float = 0.3
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
