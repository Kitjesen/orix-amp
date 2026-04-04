#!/usr/bin/env python3
"""Convert motion_imitation 19-col format to orix amp-rsl-rl .npy format.

motion_imitation format (19 floats/frame):
  [0:3]   root_pos (x, y, z)
  [3:7]   root_rot (qx, qy, qz, qw) — PyBullet convention
  [7:19]  joint_pos (12) — Laikago/A1 order

Conversion:
1. Reorder joints: PyBullet (FR,FL,RR,RL) → Isaac (FL,FR,RL,RR)
2. Negate right-side thigh/calf (orix mirrored axes)
3. Clamp to orix URDF limits
4. Scale positions by height ratio
5. Compute velocities via finite differences
6. Output amp-rsl-rl .npy format
"""

import json
import numpy as np
import os
import sys

# PyBullet → Isaac joint reorder
PB_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Height scaling
HEIGHT_RATIO = 0.28 / 0.35

# Orix joint names
ORIX_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Right-side negate indices (after reorder to Isaac)
RIGHT_NEGATE = [4, 5, 10, 11]


def convert_19col(input_path: str, output_path: str):
    """Convert one motion_imitation 19-col .txt to amp-rsl-rl .npy."""
    with open(input_path, "r") as f:
        data = json.load(f)

    frames = np.array(data["Frames"], dtype=np.float32)
    dt = data["FrameDuration"]
    fps = 1.0 / dt
    num_frames = frames.shape[0]

    # Extract
    root_pos = frames[:, 0:3].copy()
    root_rot = frames[:, 3:7].copy()  # xyzw (PyBullet)
    joint_pos_pb = frames[:, 7:19].copy()

    # 1. Reorder joints
    joint_pos = joint_pos_pb[:, PB_TO_ISAAC]

    # 2. Negate right-side
    for idx in RIGHT_NEGATE:
        joint_pos[:, idx] *= -1.0

    # 3. Clamp to orix limits
    for idx in [0, 6]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [1, 7]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 1.0)
    for idx in [2, 8]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.7, 0.0)
    for idx in [3, 9]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [4, 10]: joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.0, 0.5)
    for idx in [5, 11]: joint_pos[:, idx] = np.clip(joint_pos[:, idx], 0.0, 1.7)

    # 4. Scale root position
    root_pos *= HEIGHT_RATIO

    # 5. Build amp-rsl-rl .npy format
    # root_quaternion stays xyzw (SciPy convention, what amp-rsl-rl expects)
    amp_data = {
        "joints_list": ORIX_JOINTS,
        "joint_positions": [joint_pos[i] for i in range(num_frames)],
        "root_position": [root_pos[i] for i in range(num_frames)],
        "root_quaternion": [root_rot[i] for i in range(num_frames)],  # xyzw
        "fps": fps,
    }

    np.save(output_path, amp_data)
    print(f"  {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    print(f"    Frames: {num_frames}, FPS: {fps:.0f}, Duration: {num_frames*dt:.2f}s")
    print(f"    Joint range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "motion_imit_raw"
    output_dir = os.path.dirname(os.path.abspath(__file__))

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt files in {input_dir}")
        return

    print(f"=== Converting motion_imitation 19-col → orix .npy ===")
    print(f"  Height ratio: {HEIGHT_RATIO:.2f}")
    print()

    for txt in txt_files:
        name = os.path.splitext(txt)[0]
        convert_19col(
            os.path.join(input_dir, txt),
            os.path.join(output_dir, f"orix_{name}.npy"),
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
