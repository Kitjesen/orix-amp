# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP environment config — based on robot_lab g1_amp pattern.

Adapted from G1AmpDanceEnvCfg for a 12-DOF quadruped (orix_dog).
Uses Isaac Lab DirectRLEnv + skrl AMP algorithm.
"""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "motions")

# ── Orix Dog Robot Config ──
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # slightly above standing height
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.65,
            "FR_calf_joint": 1.5,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.65,
            "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": -0.65,
            "RR_calf_joint": 1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.65,
            "RL_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": sim_utils.DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=18.0,
            saturation_effort=18.0,
            velocity_limit=35.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


@configclass
class OrixAmpEnvCfg(DirectRLEnvCfg):
    """Orix Dog AMP environment config."""

    # ── Reward weights ──
    # Basic reward
    rew_termination = -10.0
    rew_action_l2 = -0.01
    rew_joint_pos_limits = -10.0
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    # Imitation reward
    rew_imitation_pos = 1.0
    rew_imitation_rot = 0.5
    rew_imitation_joint_pos = 2.5
    rew_imitation_joint_vel = 1.0
    imitation_sigma_pos = 1.2
    imitation_sigma_rot = 0.5
    imitation_sigma_joint_pos = 1.5
    imitation_sigma_joint_vel = 8.0

    # ── Env ──
    episode_length_s = 5.0
    decimation = 4        # physics substeps per policy step
    dt = 1 / 120          # physics dt

    # ── Spaces ──
    # policy obs: joint_pos(12) + joint_vel(12) + root_pos(3) + root_quat(4) + key_body_pos(4*3) + progress(1)
    observation_space = 12 + 12 + 3 + 4 + 4 * 3 + 1  # = 44
    action_space = 12     # 12 DOF leg joints
    state_space = 0
    num_amp_observations = 3
    # AMP obs excludes progress (not present in motion reference data): 12+12+3+4+12 = 43
    amp_observation_space = 43

    early_termination = True
    termination_height = 0.15  # lower than G1 since orix is small

    motion_file = os.path.join(MOTIONS_DIR, "orix_trot_medium_30.npz")
    reference_body = "base_link"
    reset_strategy = "random-start"

    # ── Simulation ──
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # ── Scene ──
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # ── Robot ──
    robot: ArticulationCfg = ORIX_DOG_CFG
