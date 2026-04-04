# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Orix Dog AMP — Manager-based env config for Isaac Lab + amp-rsl-rl.

Extends Go2 LocomotionVelocityRoughEnvCfg with an AMP observation group.
Standard locomotion obs (policy) + AMP discriminator obs (amp).

amp-rsl-rl AMPOnPolicyRunner expects:
  - observations["policy"]: standard locomotion obs
  - extras["observations"]["amp"]: joint_pos + joint_vel + base_lin_vel_local + base_ang_vel_local
  - env.cfg.observations.amp.joint_pos.params['asset_cfg'].joint_names
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


# ── Orix Dog Robot Config ──
ORIX_DOG_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        asset_path="/home/bsrl/hongsenpang/RLbased/robot_lab/source/robot_lab/robot_lab/data/Robots/orix_dog/urdf/orix_dog.urdf",
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
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "FR_hip_joint": 0.0, "FR_thigh_joint": -0.65, "FR_calf_joint": 1.5,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.65, "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.0, "RR_thigh_joint": -0.65, "RR_calf_joint": 1.5,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 0.65, "RL_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=18.0,
            saturation_effort=18.0,
            velocity_limit=35.0,
            stiffness=12.5,
            damping=0.3,
            friction=0.0,
        ),
    },
)

# ── Joint names for AMP obs (must match .npy joints_list) ──
ORIX_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


@configclass
class OrixAmpObservationsCfg:
    """Observations: policy (standard locomotion) + amp (for discriminator)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Standard Go2-style locomotion obs."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        """AMP discriminator obs: joint_pos + joint_vel + base_lin_vel_local + base_ang_vel_local.

        Must match amp-rsl-rl MotionData.get_amp_dataset_obs() format:
          cat(joint_positions, joint_velocities, base_lin_vel_local, base_ang_vel_local)
        """
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ORIX_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ORIX_JOINT_NAMES, preserve_order=True)},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # body frame
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # body frame

        def __post_init__(self):
            self.enable_corruption = False  # AMP obs must be clean (no noise)
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class OrixAmpRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Orix Dog AMP environment — Manager-based, for amp-rsl-rl training."""

    observations: OrixAmpObservationsCfg = OrixAmpObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = ORIX_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Scale down terrain for small robot
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Actions: 12 DOF position control
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = ORIX_JOINT_NAMES

        # Commands: conservative
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Events
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.5)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards (Go2 standard + tuned for small robot)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # Disable height scan (flat terrain first, simpler)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
