# Copyright (c) 2026 Inovxio
# SPDX-License-Identifier: Apache-2.0
"""Quadruped-specific motion loader for AMP training.

Based on robot_lab's MotionLoader but simplified for 12-DOF quadrupeds.
Handles joint name mapping between motion data and robot URDF.
"""

import os
import numpy as np
import torch


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation for batched quaternions.

    Args:
        q0: Start quaternions (..., 4)
        q1: End quaternions (..., 4)
        t:  Blend weight (..., 1) broadcastable to q0/q1 shape

    Returns:
        Interpolated unit quaternions (..., 4)
    """
    # Handle double-cover: q and -q represent the same rotation;
    # pick the sign that gives the shortest arc.
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # (..., 1)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(0.0, 1.0)

    theta = torch.acos(dot)          # (..., 1)
    sin_theta = torch.sin(theta)     # (..., 1)
    near_zero = sin_theta.abs() < 1e-6

    # SLERP weights; fall back to lerp when quaternions are nearly identical
    w0 = torch.where(near_zero, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    w1 = torch.where(near_zero, t,        torch.sin(t * theta) / sin_theta)

    result = w0 * q0 + w1 * q1
    # Re-normalise to guard against floating-point drift
    return result / result.norm(dim=-1, keepdim=True).clamp(min=1e-8)


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
        print(f"Motion loaded ({os.path.basename(motion_file)}): {self.duration:.1f}s, "
              f"{self.num_frames} frames, {len(self._dof_names)} DOFs")

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
                print(f"[WARN] Body '{name}' not in motion data, using index 0")
                indices.append(0)
        return indices

    def sample(
        self, num_samples: int, times: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        """Sample motion data at specified times.

        Args:
            num_samples: Number of samples (= num_envs).
            times: Array of times in seconds, shape (num_samples,).

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
        blend_1d = blend.unsqueeze(-1)        # (N, 1)  — for scalar/vector lerp

        def lerp(data: torch.Tensor, dim_extra: int = 0) -> torch.Tensor:
            """Linear interpolation with optional extra broadcast dims."""
            low = data[frame_low]
            high = data[frame_high]
            b = blend_1d
            for _ in range(dim_extra):
                b = b.unsqueeze(-1)
            return low + b * (high - low)

        # For body rotations use SLERP so intermediate quats stay on the unit sphere
        # blend shape for (N, num_bodies, 4): (N, 1, 1)
        blend_3d = blend_1d.unsqueeze(-1)     # (N, 1, 1)
        body_rot_interp = _slerp(
            self.body_rotations[frame_low],   # (N, num_bodies, 4)
            self.body_rotations[frame_high],  # (N, num_bodies, 4)
            blend_3d,                         # (N, 1, 1)
        )

        return (
            lerp(self.dof_positions),            # (N, num_dofs)
            lerp(self.dof_velocities),            # (N, num_dofs)
            lerp(self.body_positions, 1),         # (N, num_bodies, 3)
            body_rot_interp,                      # (N, num_bodies, 4) — SLERP
            lerp(self.body_linear_velocities, 1),
            lerp(self.body_angular_velocities, 1),
        )
