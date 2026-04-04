#!/usr/bin/env python3
"""Convert A1 AMP motion data (.txt JSON) to amp-rsl-rl .npy format.

amp-rsl-rl expects .npy dict with:
  - joints_list: List[str] — joint names
  - joint_positions: List[np.ndarray] — per-frame joint angles
  - root_position: List[np.ndarray] — per-frame base XYZ
  - root_quaternion: List[np.ndarray] — per-frame quat in xyzw (SciPy convention)
  - fps: float

Source: AMP_for_hardware .txt JSON (61 floats/frame)
  [0:3] root_pos, [3:7] root_rot(xyzw), [7:19] joint_pos(12, PyBullet order)

Retargeting for orix_dog:
  - Reorder: PyBullet (FR,FL,RR,RL) → Isaac (FL,FR,RL,RR)
  - Right-side thigh/calf: negate (orix has mirrored axis)
  - Scale positions by 0.8 (height ratio A1→orix)
"""

import json
import numpy as np
import os
import sys


# PyBullet → Isaac joint reorder
PB_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Height scaling
HEIGHT_RATIO = 0.28 / 0.35  # orix / A1

# Orix joint names (Isaac order)
ORIX_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Right-side indices that need negation (axis mirrored in orix URDF)
RIGHT_NEGATE = [4, 5, 10, 11]  # FR_thigh, FR_calf, RR_thigh, RR_calf


def convert_a1_txt_to_amp_npy(input_path: str, output_path: str):
    """Convert one A1 .txt motion file to amp-rsl-rl .npy format."""
    with open(input_path, "r") as f:
        data = json.load(f)

    frames = np.array(data["Frames"], dtype=np.float32)
    fps = 1.0 / data["FrameDuration"]
    num_frames = frames.shape[0]

    # Extract fields
    root_pos = frames[:, 0:3].copy()       # (N, 3) world position
    root_rot = frames[:, 3:7].copy()       # (N, 4) quaternion xyzw (PyBullet)
    joint_pos_pb = frames[:, 7:19].copy()  # (N, 12) PyBullet order

    # 1. Reorder joints to Isaac convention
    joint_pos = joint_pos_pb[:, PB_TO_ISAAC]

    # 2. Negate right-side joints (orix mirrored axes: FR/RR thigh/calf axis +y)
    for idx in RIGHT_NEGATE:
        joint_pos[:, idx] *= -1.0

    # 3. Clamp to orix URDF limits per side
    # Left (FL/RL): hip[-0.5,0.5], thigh[-0.5,1.0], calf[-1.7,0]
    for idx in [0, 6]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [1, 7]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 1.0)
    for idx in [2, 8]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.7, 0.0)
    # Right (FR/RR, negated): thigh[-1.0,0.5], calf[0,1.7]
    for idx in [3, 9]:  joint_pos[:, idx] = np.clip(joint_pos[:, idx], -0.5, 0.5)
    for idx in [4, 10]: joint_pos[:, idx] = np.clip(joint_pos[:, idx], -1.0, 0.5)
    for idx in [5, 11]: joint_pos[:, idx] = np.clip(joint_pos[:, idx], 0.0, 1.7)

    # 4. Scale root position
    root_pos *= HEIGHT_RATIO

    # 4. Build amp-rsl-rl format: lists of per-frame arrays
    joint_positions_list = [joint_pos[i] for i in range(num_frames)]
    root_position_list = [root_pos[i] for i in range(num_frames)]
    root_quaternion_list = [root_rot[i] for i in range(num_frames)]  # already xyzw

    amp_data = {
        "joints_list": ORIX_JOINTS,
        "joint_positions": joint_positions_list,
        "root_position": root_position_list,
        "root_quaternion": root_quaternion_list,  # xyzw (SciPy/PyBullet convention)
        "fps": fps,
    }

    np.save(output_path, amp_data)
    print(f"  {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    print(f"    Frames: {num_frames}, FPS: {fps:.0f}, Duration: {num_frames/fps:.1f}s")
    print(f"    Joint range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "a1_raw"
    output_dir = os.path.dirname(os.path.abspath(__file__))

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt files in {input_dir}")
        return

    print(f"=== Converting A1 → orix amp-rsl-rl .npy ===")
    print(f"  Height ratio: {HEIGHT_RATIO:.2f}")
    print()

    for txt in txt_files:
        name = os.path.splitext(txt)[0]
        convert_a1_txt_to_amp_npy(
            os.path.join(input_dir, txt),
            os.path.join(output_dir, f"orix_{name}.npy"),
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
