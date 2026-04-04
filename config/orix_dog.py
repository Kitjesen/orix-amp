"""Configuration for Orix Dog quadruped robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

_URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "urdf", "orix_dog.urdf")

ORIX_DOG_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.32),
        joint_pos={
            ".*_hip_joint": 0.0,
            # Left legs (FL, RL): thigh limits [-0.5, 1.0], calf limits [-1.7, 0.0]
            # Use 0.65 (not 0.8) to keep 0.35 rad margin from limits
            ".*L_thigh_joint":  0.65,
            ".*L_calf_joint":  -1.3,
            # Right legs (FR, RR): URDF axis mirrored, so opposite sign
            ".*R_thigh_joint": -0.65,
            ".*R_calf_joint":   1.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=23.7,
            saturation_effort=23.7,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
