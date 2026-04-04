#!/usr/bin/env python3
"""Retarget A1 AMP motion data to orix_dog.

A1 and orix_dog have identical joint topology (12 DOF, same joint names).
Retargeting only needs:
1. Scale root height (A1: 0.35m → orix: 0.28m, ratio=0.8)
2. Scale root XY positions by same ratio
3. Clamp joint angles to orix limits
4. Scale velocities accordingly

Input: AMP_for_hardware JSON .txt format
Output: robot_lab .npz format (for Isaac Lab) + .txt format (for legged_gym)

A1 .txt frame format (61 floats per frame):
  [0:3]   root_pos (x, y, z)
  [3:7]   root_rot (qx, qy, qz, qw) — PyBullet convention
  [7:19]  joint_pos (12) — PyBullet order: FR, FL, RR, RL
  [19:31] toe_pos_local (12) — 4 toes × 3D
  [31:34] linear_vel (3)
  [34:37] angular_vel (3)
  [37:49] joint_vel (12) — same order as joint_pos
  [49:61] toe_vel_local (12)
"""

import json
import numpy as np
import os
import sys


# ── Robot dimensions ──
A1_STANDING_HEIGHT = 0.35
ORIX_STANDING_HEIGHT = 0.28
HEIGHT_RATIO = ORIX_STANDING_HEIGHT / A1_STANDING_HEIGHT  # 0.8

# ── Orix joint limits (from URDF) ──
ORIX_JOINT_LIMITS = {
    # (lower, upper) in radians
    "hip": (-0.5, 0.5),
    "thigh": (-0.5, 1.0),    # front legs
    "calf": (-1.7, 0.0),     # front legs
    "thigh_rear": (-1.0, 0.5),  # rear legs (flipped)
    "calf_rear": (0.0, 1.7),    # rear legs (flipped)
}

# ── Joint reorder: PyBullet (FR,FL,RR,RL) → Isaac (FL,FR,RL,RR) ──
PYBULLET_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # index mapping


def load_a1_motion_txt(filepath: str) -> dict:
    """Load AMP_for_hardware .txt motion file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    frames = np.array(data["Frames"], dtype=np.float32)
    fps = int(round(1.0 / data["FrameDuration"]))
    loop_mode = data.get("LoopMode", "Wrap")

    print(f"  Loaded: {os.path.basename(filepath)}")
    print(f"  Frames: {frames.shape[0]}, FPS: {fps}, Duration: {frames.shape[0]/fps:.1f}s")
    print(f"  LoopMode: {loop_mode}")

    return {
        "frames": frames,
        "fps": fps,
        "frame_duration": data["FrameDuration"],
        "loop_mode": loop_mode,
        "weight": data.get("MotionWeight", 1.0),
    }


def reorder_joints(joint_data: np.ndarray) -> np.ndarray:
    """Reorder from PyBullet (FR,FL,RR,RL) to Isaac (FL,FR,RL,RR)."""
    return joint_data[:, PYBULLET_TO_ISAAC]


def retarget_to_orix(a1_data: dict) -> dict:
    """Retarget A1 motion to orix_dog.

    Returns dict with both Isaac-order joint data and body data.
    """
    frames = a1_data["frames"]
    num_frames = frames.shape[0]
    fps = a1_data["fps"]

    # ── Extract fields ──
    root_pos = frames[:, 0:3].copy()
    root_rot = frames[:, 3:7].copy()  # qx, qy, qz, qw (PyBullet)
    joint_pos_pb = frames[:, 7:19].copy()
    toe_pos_local = frames[:, 19:31].copy()
    lin_vel = frames[:, 31:34].copy()
    ang_vel = frames[:, 34:37].copy()
    joint_vel_pb = frames[:, 37:49].copy()
    toe_vel_local = frames[:, 49:61].copy()

    # ── 1. Reorder joints to Isaac convention ──
    joint_pos = reorder_joints(joint_pos_pb)
    joint_vel = reorder_joints(joint_vel_pb)

    # ── 2. Handle right-side mirrored axes ──
    # A1 URDF: all thigh/calf use axis -y
    # orix URDF: FR/RR thigh/calf use axis +y (mirrored, intentional design)
    # Isaac Lab reads axis from URDF, so positive joint value on +y axis
    # produces opposite physical motion compared to -y axis.
    # Therefore: right-side thigh/calf values must be NEGATED.
    # (Negation is done below in the clamp section, lines 138-145)

    # ── 3. Scale root position ──
    root_pos[:, 2] *= HEIGHT_RATIO  # scale height
    # XY: scale by same ratio (smaller stride for smaller robot)
    root_pos[:, 0] *= HEIGHT_RATIO
    root_pos[:, 1] *= HEIGHT_RATIO

    # ── 3. Scale linear velocity ──
    lin_vel *= HEIGHT_RATIO

    # ── 4. Scale toe positions ──
    toe_pos_local *= HEIGHT_RATIO

    # ── 6. Clamp joint angles to orix URDF limits (per-side) ──
    # Isaac order: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
    # FL/RL (axis -y): thigh [-0.5, 1.0], calf [-1.7, 0]
    # FR/RR (axis +y): thigh [-1.0, 0.5], calf [0, 1.7]
    # A1 values are all in "left convention" (thigh~0.8, calf~-1.5)
    # For right-side: A1 value 0.8 maps to orix FR_thigh range [-1.0, 0.5]
    # Need to NEGATE for right side to get correct physical direction

    # Left side: clamp to left limits directly
    for idx in [0, 6]:  # FL_hip, RL_hip
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [1, 7]:  # FL_thigh, RL_thigh
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 1.0)
    for idx in [2, 8]:  # FL_calf, RL_calf
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.7, 0.0)

    # Right side: negate (axis flipped), then clamp to right limits
    for idx in [3, 9]:  # FR_hip, RR_hip
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [4, 10]:  # FR_thigh, RR_thigh — negate A1 values for +y axis
        joint_pos[:, idx] *= -1.0
        joint_vel[:, idx] *= -1.0
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.0, 0.5)
    for idx in [5, 11]:  # FR_calf, RR_calf — negate A1 values for +y axis
        joint_pos[:, idx] *= -1.0
        joint_vel[:, idx] *= -1.0
        joint_pos[:, idx] = np.clip(joint_pos[:, idx], 0.0, 1.7)

    # ── 6. Convert quaternion: PyBullet (x,y,z,w) → Isaac Lab (w,x,y,z) ──
    root_rot_wxyz = np.zeros_like(root_rot)
    root_rot_wxyz[:, 0] = root_rot[:, 3]  # w
    root_rot_wxyz[:, 1] = root_rot[:, 0]  # x
    root_rot_wxyz[:, 2] = root_rot[:, 1]  # y
    root_rot_wxyz[:, 3] = root_rot[:, 2]  # z

    # ── Build output ──
    dof_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    body_names = ["base_link", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Body positions: base + 4 feet from toe_pos_local
    body_positions = np.zeros((num_frames, 5, 3), dtype=np.float32)
    body_positions[:, 0] = root_pos
    # Reorder toes: PyBullet (FR,FL,RR,RL) → Isaac (FL,FR,RL,RR)
    toe_reorder = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    toe_pos_reordered = toe_pos_local[:, toe_reorder].reshape(num_frames, 4, 3)
    for i in range(4):
        body_positions[:, i + 1] = root_pos + toe_pos_reordered[:, i]  # already scaled at line 117

    # Body rotations: base = root_rot, feet = identity
    body_rotations = np.zeros((num_frames, 5, 4), dtype=np.float32)
    body_rotations[:, 0] = root_rot_wxyz
    body_rotations[:, 1:, 0] = 1.0  # identity quaternion (w=1)

    # Body velocities
    body_lin_vel = np.zeros((num_frames, 5, 3), dtype=np.float32)
    body_lin_vel[:, 0] = lin_vel
    body_ang_vel = np.zeros((num_frames, 5, 3), dtype=np.float32)
    body_ang_vel[:, 0] = ang_vel

    return {
        "fps": np.int64(fps),
        "dof_names": np.array(dof_names),
        "body_names": np.array(body_names),
        "dof_positions": joint_pos,
        "dof_velocities": joint_vel,
        "body_positions": body_positions,
        "body_rotations": body_rotations,
        "body_linear_velocities": body_lin_vel,
        "body_angular_velocities": body_ang_vel,
        # Keep original for legged_gym format
        "_raw_frames_isaac": np.column_stack([
            root_pos, root_rot,  # keep PyBullet quat for legged_gym
            joint_pos, toe_pos_local,
            lin_vel, ang_vel,
            joint_vel, toe_vel_local,
        ]),
    }


def save_npz(data: dict, output_path: str):
    """Save in robot_lab MotionLoader .npz format."""
    np.savez(
        output_path,
        fps=data["fps"],
        dof_names=data["dof_names"],
        body_names=data["body_names"],
        dof_positions=data["dof_positions"],
        dof_velocities=data["dof_velocities"],
        body_positions=data["body_positions"],
        body_rotations=data["body_rotations"],
        body_linear_velocities=data["body_linear_velocities"],
        body_angular_velocities=data["body_angular_velocities"],
    )
    print(f"  Saved NPZ: {output_path}")


def save_legged_gym_txt(data: dict, output_path: str, frame_duration: float, loop_mode: str = "Wrap"):
    """Save in AMP_for_hardware .txt format (for legged_gym WMP)."""
    frames = data["_raw_frames_isaac"]
    motion_data = {
        "LoopMode": loop_mode,
        "FrameDuration": frame_duration,
        "MotionWeight": 1.0,
        "Frames": frames.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(motion_data, f)
    print(f"  Saved TXT: {output_path}")


def main():
    # Input: A1 motion files (from server or local)
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = os.path.dirname(os.path.abspath(__file__))

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt motion files found in {input_dir}")
        print("Usage: python retarget_a1_to_orix.py <path_to_a1_motion_dir>")
        return

    print(f"=== Retargeting A1 → orix_dog ===")
    print(f"  Height ratio: {HEIGHT_RATIO:.2f} (A1: {A1_STANDING_HEIGHT}m → orix: {ORIX_STANDING_HEIGHT}m)")
    print(f"  Input: {input_dir} ({len(txt_files)} files)")
    print()

    for txt_file in txt_files:
        print(f"Processing: {txt_file}")
        a1_data = load_a1_motion_txt(os.path.join(input_dir, txt_file))
        orix_data = retarget_to_orix(a1_data)

        name = os.path.splitext(txt_file)[0]
        # Save in both formats
        save_npz(orix_data, os.path.join(output_dir, f"orix_{name}.npz"))
        save_legged_gym_txt(
            orix_data,
            os.path.join(output_dir, f"orix_{name}.txt"),
            frame_duration=a1_data["frame_duration"],
        )
        print()

    print("=== Done! Retargeted motion files ready. ===")


if __name__ == "__main__":
    main()
