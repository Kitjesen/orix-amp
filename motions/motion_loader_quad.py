# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Quadruped-specific motion loader for AMP training.

Based on robot_lab's MotionLoader but simplified for 12-DOF quadrupeds.
Handles joint name mapping between motion data and robot URDF.
"""

import os
import numpy as np
import torch


class QuadMotionLoader:
    """Load and sample quadruped motion reference data."""

    def __init__(self, motion_file: str, device: torch.device) -> None:
        assert os.path.isfile(motion_file), f"Invalid file: {motion_file}"
        data = np.load(motion_file, allow_pickle=True)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=device)
        self.body_linear_velocities = torch.tensor(data["body_linear_velocities"], dtype=torch.float32, device=device)
        self.body_angular_velocities = torch.tensor(data["body_angular_velocities"], dtype=torch.float32, device=device)

        self.dt = 1.0 / float(data["fps"])
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({os.path.basename(motion_file)}): {self.duration:.1f}s, {self.num_frames} frames, {len(self._dof_names)} DOFs")

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    def get_dof_index(self, robot_joint_names: list[str]) -> list[int]:
        """Map robot joint names to motion data DOF indices."""
        indices = []
        for name in robot_joint_names:
            if name in self._dof_names:
                indices.append(self._dof_names.index(name))
            else:
                print(f"[WARN] Joint '{name}' not in motion data, using index 0")
                indices.append(0)
        return indices

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Map body names to motion data body indices."""
        indices = []
        for name in body_names:
            if name in self._body_names:
                indices.append(self._body_names.index(name))
            else:
                indices.append(0)  # fallback to first body
        return indices

    def sample(
        self, num_samples: int, times: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        """Sample motion data at specified times.

        Args:
            num_samples: Number of samples (= num_envs).
            times: Array of times in seconds.

        Returns:
            Tuple of (dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)
        """
        # Wrap times to motion duration (loop)
        times = np.mod(times, self.duration)

        # Convert to frame indices
        frame_float = times / self.dt
        frame_low = np.clip(np.floor(frame_float).astype(int), 0, self.num_frames - 2)
        frame_high = frame_low + 1
        blend = torch.tensor(frame_float - frame_low, dtype=torch.float32, device=self.device)
        blend = blend.unsqueeze(-1)  # (N, 1) for broadcasting

        # Linear interpolation
        def lerp(data: torch.Tensor, dim_extra: int = 0) -> torch.Tensor:
            low = data[frame_low]
            high = data[frame_high]
            b = blend
            for _ in range(dim_extra):
                b = b.unsqueeze(-1)
            return low + b * (high - low)

        return (
            lerp(self.dof_positions),           # (N, num_dofs)
            lerp(self.dof_velocities),           # (N, num_dofs)
            lerp(self.body_positions, 1),        # (N, num_bodies, 3)
            lerp(self.body_rotations, 1),        # (N, num_bodies, 4) — note: should slerp for quats
            lerp(self.body_linear_velocities, 1),
            lerp(self.body_angular_velocities, 1),
        )
