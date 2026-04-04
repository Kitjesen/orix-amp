#!/usr/bin/env python3
"""Generate trot gait reference motion data for orix_dog AMP training.

Creates a .npz file matching robot_lab's MotionLoader format:
  - dof_names: joint names
  - body_names: key body names
  - dof_positions: (num_frames, 12) joint angles
  - dof_velocities: (num_frames, 12) joint velocities
  - body_positions: (num_frames, num_bodies, 3) body positions
  - body_rotations: (num_frames, num_bodies, 4) body quaternions
  - body_linear_velocities: (num_frames, num_bodies, 3)
  - body_angular_velocities: (num_frames, num_bodies, 3)
  - fps: frames per second

Trot gait: diagonal legs move together.
  Phase 1: FL+RR swing, FR+RL stance
  Phase 2: FR+RL swing, FL+RR stance

Orix_dog joints (12 DOF):
  FL_hip_joint, FL_thigh_joint, FL_calf_joint
  FR_hip_joint, FR_thigh_joint, FR_calf_joint
  RL_hip_joint, RL_thigh_joint, RL_calf_joint
  RR_hip_joint, RR_thigh_joint, RR_calf_joint
"""

import numpy as np
import os


def generate_trot_motion(
    duration: float = 5.0,
    fps: int = 30,
    stride_freq: float = 2.0,     # Hz (strides per second)
    forward_speed: float = 0.5,    # m/s
    base_height: float = 0.28,     # m (orix standing height)
    swing_height: float = 0.04,    # m (foot lift)
) -> dict:
    """Generate a trot gait motion dataset.

    Trot: diagonal pair (FL+RR) alternates with (FR+RL).
    """
    num_frames = int(duration * fps)
    dt = 1.0 / fps
    t = np.linspace(0, duration, num_frames)

    # ── Joint names (must match URDF order) ──
    dof_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    # ── Body names for AMP discriminator ──
    body_names = [
        "base_link",
        "FL_foot", "FR_foot", "RL_foot", "RR_foot",
    ]
    num_bodies = len(body_names)

    # ── Default standing pose (radians) — must match init_state joint_pos ──
    # Front legs (FL/FR): thigh=+0.65, calf=-1.5
    # Rear legs (RL/RR): thigh=-0.65, calf=+1.5
    default_hip = 0.0
    default_thigh = 0.65   # magnitude for front legs (positive = forward)
    default_calf = -1.5    # front legs calf (negative = bent down)

    # ── Generate trot gait ──
    dof_positions = np.zeros((num_frames, 12), dtype=np.float32)

    # Phase for each leg: FL=0, FR=pi, RL=pi, RR=0 (trot pattern)
    phase_offsets = [0.0, np.pi, np.pi, 0.0]  # FL, FR, RL, RR

    for frame in range(num_frames):
        time = t[frame]
        cycle_phase = 2 * np.pi * stride_freq * time

        for leg_idx in range(4):
            hip_idx = leg_idx * 3
            thigh_idx = leg_idx * 3 + 1
            calf_idx = leg_idx * 3 + 2

            leg_phase = cycle_phase + phase_offsets[leg_idx]

            # Hip: slight lateral sway
            hip_swing = 0.05 * np.sin(leg_phase)

            # Thigh: forward/backward swing for locomotion
            thigh_swing = 0.3 * np.sin(leg_phase)

            # Calf: lift during swing phase (more bent when swinging)
            # swing phase: sin > 0, stance phase: sin < 0
            swing_factor = max(0, np.sin(leg_phase))
            calf_swing = -0.3 * swing_factor  # more bent during swing

            # Front legs: thigh positive = forward
            # Rear legs: thigh sign flipped for rear locomotion
            if leg_idx >= 2:  # rear legs (RL, RR): thigh=-0.65, calf=+1.5
                thigh_sign = -1.0
                calf_default = 1.5  # rear legs calf is positive
            else:  # front legs (FL, FR): thigh=+0.65, calf=-1.5
                thigh_sign = 1.0
                calf_default = -1.5

            dof_positions[frame, hip_idx] = default_hip + hip_swing
            dof_positions[frame, thigh_idx] = thigh_sign * (default_thigh + thigh_swing)
            dof_positions[frame, calf_idx] = calf_default + calf_swing * thigh_sign

    # ── Compute velocities via finite differences ──
    dof_velocities = np.zeros_like(dof_positions)
    dof_velocities[1:] = (dof_positions[1:] - dof_positions[:-1]) / dt
    dof_velocities[0] = dof_velocities[1]

    # ── Body positions and rotations ──
    body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_linear_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_angular_velocities = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

    # Base link: moves forward at constant speed, slight vertical oscillation
    for frame in range(num_frames):
        time = t[frame]
        cycle_phase = 2 * np.pi * stride_freq * time

        # Base position: forward motion + slight vertical bounce
        body_positions[frame, 0, 0] = forward_speed * time  # x forward
        body_positions[frame, 0, 1] = 0.0                    # y lateral
        body_positions[frame, 0, 2] = base_height + 0.005 * np.sin(2 * cycle_phase)  # z with bounce

        # Base rotation: identity quaternion (w, x, y, z) with slight pitch oscillation
        pitch = 0.02 * np.sin(cycle_phase)
        body_rotations[frame, 0] = [np.cos(pitch/2), 0, np.sin(pitch/2), 0]  # wxyz

        # Foot positions (approximate, for discriminator)
        # FL
        body_positions[frame, 1] = body_positions[frame, 0] + [0.15, 0.1, -base_height]
        # FR
        body_positions[frame, 2] = body_positions[frame, 0] + [0.15, -0.1, -base_height]
        # RL
        body_positions[frame, 3] = body_positions[frame, 0] + [-0.15, 0.1, -base_height]
        # RR
        body_positions[frame, 4] = body_positions[frame, 0] + [-0.15, -0.1, -base_height]

        # Foot swing height
        for foot_idx, phase_off in enumerate(phase_offsets):
            leg_phase = cycle_phase + phase_off
            swing = max(0, np.sin(leg_phase))
            body_positions[frame, foot_idx + 1, 2] += swing_height * swing

        # Foot rotations: identity
        for b in range(num_bodies):
            if body_rotations[frame, b, 0] == 0:
                body_rotations[frame, b] = [1, 0, 0, 0]

    # Body velocities via finite differences
    body_linear_velocities[1:] = (body_positions[1:] - body_positions[:-1]) / dt
    body_linear_velocities[0] = body_linear_velocities[1]

    return {
        "fps": np.int64(fps),
        "dof_names": np.array(dof_names),
        "body_names": np.array(body_names),
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_linear_velocities,
        "body_angular_velocities": body_angular_velocities,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate trot at different speeds
    for name, speed, freq in [
        ("trot_slow", 0.3, 1.5),
        ("trot_medium", 0.5, 2.0),
        ("trot_fast", 0.8, 2.5),
    ]:
        print(f"Generating {name}...")
        data = generate_trot_motion(
            duration=5.0, fps=30,
            stride_freq=freq, forward_speed=speed,
        )
        path = os.path.join(out_dir, f"orix_{name}_30.npz")
        np.savez(path, **data)
        print(f"  Saved: {path}")
        print(f"  Frames: {data['dof_positions'].shape[0]}, DOFs: {data['dof_positions'].shape[1]}")
        print(f"  Duration: {data['dof_positions'].shape[0] / data['fps']:.1f}s")
        print()

    print("Done! Motion files ready for AMP training.")


if __name__ == "__main__":
    main()
