#!/usr/bin/env python3
"""Generate orix_dog AMP motion data in TienKung 36-col format.

TienKung AMPLoader expects: joint_pos(N) + joint_vel(N) + end_effector_pos(M)
For orix 12-DOF quadruped: joint_pos(12) + joint_vel(12) + foot_pos_local(12) = 36 cols

Source: A1 AMP_for_hardware 61-col .txt files (already retargeted)
Output: 36-col .txt files for TienKung-style AMPLoader
"""

import json
import numpy as np
import os
import sys

# PyBullet → Isaac joint reorder
PB_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Height scaling
HEIGHT_RATIO = 0.28 / 0.35

# Right-side negate (orix mirrored axes)
RIGHT_NEGATE = [4, 5, 10, 11]


def convert_61col_to_36col(input_path: str, output_path: str):
    """Convert AMP_for_hardware 61-col to TienKung-style 36-col."""
    with open(input_path, "r") as f:
        data = json.load(f)

    frames_raw = np.array(data["Frames"], dtype=np.float32)
    dt = data["FrameDuration"]
    num_frames = frames_raw.shape[0]

    # Extract from 61-col
    joint_pos_pb = frames_raw[:, 7:19].copy()   # 12 joints (PyBullet order)
    joint_vel_pb = frames_raw[:, 37:49].copy()   # 12 joint velocities
    toe_pos_local = frames_raw[:, 19:31].copy()  # 4 toes × 3D (local frame)

    # Reorder to Isaac convention
    joint_pos = joint_pos_pb[:, PB_TO_ISAAC]
    joint_vel = joint_vel_pb[:, PB_TO_ISAAC]
    toe_reorder = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    foot_pos = toe_pos_local[:, toe_reorder]

    # Negate right-side (orix mirrored axes)
    for idx in RIGHT_NEGATE:
        joint_pos[:, idx] *= -1.0
        joint_vel[:, idx] *= -1.0

    # Scale foot positions
    foot_pos *= HEIGHT_RATIO

    # Build 36-col: joint_pos(12) + joint_vel(12) + foot_pos_local(12)
    frames_36 = np.concatenate([joint_pos, joint_vel, foot_pos], axis=1)

    # Save as TienKung-style JSON .txt
    output_data = {
        "LoopMode": data.get("LoopMode", "Wrap"),
        "FrameDuration": dt,
        "Frames": frames_36.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"  {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    print(f"    Frames: {num_frames}, Cols: {frames_36.shape[1]}, FPS: {1/dt:.0f}")


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "amp_hw_raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt files in {input_dir}")
        return

    print(f"=== Converting 61-col → 36-col (TienKung format) ===")
    print()

    for txt in txt_files:
        name = os.path.splitext(txt)[0]
        convert_61col_to_36col(
            os.path.join(input_dir, txt),
            os.path.join(output_dir, f"orix_amp_{name}.txt"),
        )
        print()

    # Also convert from WMP data (trot1/trot2/hop1/hop2)
    wmp_dir = "a1_raw"
    if os.path.isdir(wmp_dir):
        print("=== Also converting a1_raw ===")
        for txt in sorted(os.listdir(wmp_dir)):
            if txt.endswith(".txt"):
                name = os.path.splitext(txt)[0]
                convert_61col_to_36col(
                    os.path.join(wmp_dir, txt),
                    os.path.join(output_dir, f"orix_amp_{name}.txt"),
                )
                print()

    print("Done!")


if __name__ == "__main__":
    main()
