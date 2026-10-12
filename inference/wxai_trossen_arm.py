"""
wxai_trossen_arm.py

Human2Robot RobotArm adapter for Trossen WidowX AI (WXAI) follower robot in Isaac Sim.
Implements the RobotArm Protocol from inference/interfaces.py, bridging
Human2Robot's trajectory controller and Cartesian targets to TrossenAIController.
"""

from __future__ import annotations

from typing import Optional, Protocol, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from inference.robot_constants import (
    GRIPPER_CLOSED_POSITION,
    GRIPPER_OPEN_POSITION,
)

if TYPE_CHECKING:
    from inference.wxai_controller import TrossenAIController


class GraspAssist(Protocol):
    """Optional physical-grasp bridge notified by gripper commands."""

    def update(self, commanded_closed: bool) -> None:
        ...

    def release(self) -> None:
        ...


class IsaacWXAITrossenArm:
    """
    Adapter implementing the Human2Robot RobotArm protocol for Trossen WidowX AI
    in Isaac Sim.

    Uses bounded differential IK and the native parallel jaw articulation.
    """

    def __init__(
        self,
        controller: TrossenAIController,
        simulation_app,
        T_world_to_cam: np.ndarray,
        T_base_in_cam: Optional[np.ndarray] = None,
        dt: float = 1.0 / 60.0,
        grasp_assist: Optional[GraspAssist] = None,
    ):
        """
        Args:
            controller: Initialized TrossenAIController instance.
            simulation_app: Isaac Sim SimulationApp instance.
            T_world_to_cam: 4x4 transform from USD world frame to optical camera.
            T_base_in_cam: 4x4 hand-eye calibration matrix (base -> camera).
                           If None, computed from T_world_to_cam @ T_base_in_world.
            dt: Simulation physics time step (default 1/60s).
            grasp_assist: Optional contact-gated physical attachment manager.
        """
        self.controller = controller
        self.simulation_app = simulation_app
        self.T_world_to_cam = np.asarray(T_world_to_cam, dtype=np.float64)
        self.T_cam_to_world = np.linalg.inv(self.T_world_to_cam)
        self.dt = dt
        self.grasp_assist = grasp_assist
        self.step_callback = None
        self.commanded_gripper_closed = 0.0

        # Retrieve robot base pose in world to compute or verify T_base_in_cam
        base_pos, base_quat = self.controller.get_world_poses()
        base_pos = np.asarray(base_pos.numpy(), dtype=np.float64)[0]
        base_quat = np.asarray(base_quat.numpy(), dtype=np.float64)[0]  # [w, x, y, z]

        T_base_in_world = np.eye(4, dtype=np.float64)
        T_base_in_world[:3, :3] = Rotation.from_quat(
            [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
        ).as_matrix()
        T_base_in_world[:3, 3] = base_pos
        self.T_base_in_world = T_base_in_world

        if T_base_in_cam is not None:
            self._T_base_in_cam = np.asarray(T_base_in_cam, dtype=np.float64)
        else:
            self._T_base_in_cam = self.T_world_to_cam @ self.T_base_in_world

        self.gripper_open_pos = float(GRIPPER_OPEN_POSITION)
        self.gripper_closed_pos = float(GRIPPER_CLOSED_POSITION)

    @property
    def T_base_in_cam(self) -> np.ndarray:
        """Hand-eye extrinsic: robot base -> camera optical frame."""
        return self._T_base_in_cam

    def get_T_ee_in_world(self) -> np.ndarray:
        """Get current end-effector pose in USD world frame."""
        _, ee_pos, ee_quat = self.controller.get_current_state()
        pos = ee_pos[0]  # (3,)
        quat_wxyz = ee_quat[0]  # [w, x, y, z]

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        ).as_matrix()
        T[:3, 3] = pos
        return T

    def get_T_ee_in_base(self) -> np.ndarray:
        """Current end-effector pose in robot base frame."""
        T_ee_world = self.get_T_ee_in_world()
        return np.linalg.inv(self.T_base_in_world) @ T_ee_world

    def get_T_ee_in_cam(self) -> np.ndarray:
        """Current end-effector pose in Human2Robot camera optical frame (4x4)."""
        T_ee_world = self.get_T_ee_in_world()
        T_ee_cam = self.T_world_to_cam @ T_ee_world
        return T_ee_cam.astype(np.float32)

    def move_ee_in_cam(
        self,
        T_ee_in_cam: np.ndarray,
        duration: float,
        blocking: bool = False,
    ) -> bool:
        """
        Servo the end effector to a Cartesian target in camera optical frame.
        IK is solved internally via TrossenAIController differential IK.

        Args:
            T_ee_in_cam: 4x4 target SE(3) pose in camera frame.
            duration: Commanded duration for the step in seconds.
            blocking: If True, steps simulation for duration.

        Returns:
            bool: True if command accepted and finite.
        """
        T_target_cam = np.asarray(T_ee_in_cam, dtype=np.float64)
        if not np.isfinite(T_target_cam).all():
            raise RuntimeError("Non-finite Cartesian target received in move_ee_in_cam")

        # Camera frame -> World frame
        T_target_world = self.T_cam_to_world @ T_target_cam

        target_pos = T_target_world[:3, 3]
        target_rot = Rotation.from_matrix(T_target_world[:3, :3])
        target_quat_xyzw = target_rot.as_quat()
        target_quat_wxyz = np.array(
            [
                target_quat_xyzw[3],
                target_quat_xyzw[0],
                target_quat_xyzw[1],
                target_quat_xyzw[2],
            ],
            dtype=np.float64,
        )

        n_steps = max(1, int(round(float(duration) / self.dt))) if blocking else 1
        for _ in range(n_steps):
            if not self.simulation_app.is_running():
                raise RuntimeError("Simulation closed during learned trajectory execution")
            # Recompute the bounded differential IK from live physics state each tick.
            self.controller.set_end_effector_pose(
                position=target_pos.reshape(1, 3),
                orientation=target_quat_wxyz.reshape(1, 4),
            )
            self.advance()

        return True

    def get_gripper(self) -> float:
        """
        Current gripper state normalized to [0, 1].
        0.0 = fully OPEN (0.044 m), 1.0 = fully CLOSED (0.022 m).
        """
        dof_pos = self.controller.get_dof_positions().numpy()[0]
        width = float(dof_pos[self.controller.gripper_dof_index])

        # Width ranges from 0.022 (closed) to 0.044 (open)
        norm_closed = (self.gripper_open_pos - width) / (
            self.gripper_open_pos - self.gripper_closed_pos
        )
        return float(np.clip(norm_closed, 0.0, 1.0))

    def set_gripper(self, value: float, blocking: bool = False) -> None:
        """
        Command gripper position: 0.0 = open, 1.0 = closed.

        Args:
            value: Normalized scalar in [0, 1].
            blocking: If True, steps simulation to allow physical closure.
        """
        norm_val = float(np.clip(value, 0.0, 1.0))
        if not np.isfinite(norm_val):
            raise ValueError("Gripper command must be finite")
        self.commanded_gripper_closed = norm_val
        target_width = self.gripper_open_pos - norm_val * (
            self.gripper_open_pos - self.gripper_closed_pos
        )
        self.controller.set_gripper_position(target_width)

        n_steps = 20 if blocking else 1
        for _ in range(n_steps):
            self.advance()

    def advance(self) -> None:
        """Advance one physics tick and sample physical grasp/placement state."""
        self.simulation_app.update()
        if self.grasp_assist is not None:
            self.grasp_assist.update(commanded_closed=self.commanded_gripper_closed > 0.5)
        if self.step_callback is not None:
            self.step_callback()

    def open_gripper(self, blocking: bool = False) -> None:
        """Fully open the gripper."""
        self.set_gripper(0.0, blocking=blocking)

    def close_gripper(self, blocking: bool = False) -> None:
        """Fully close the gripper."""
        self.set_gripper(1.0, blocking=blocking)

    def go_home(self, blocking: bool = True) -> None:
        """Move arm to default home joint configuration."""
        self.controller.reset_to_default_pose()
        n_steps = 60 if blocking else 1
        for _ in range(n_steps):
            self.simulation_app.update()

    def close(self) -> None:
        """Resource cleanup."""
        if self.grasp_assist is not None:
            self.grasp_assist.release()
