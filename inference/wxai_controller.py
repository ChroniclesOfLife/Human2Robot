# Copyright 2026 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Trossen AI Robot Controller.

This module provides a controller for Trossen AI robots with
differential inverse kinematics and gripper control.
Supports:
- WidowX AI
- Stationary AI
- Mobile AI
"""

from enum import Enum
from typing import Any, Callable

import numpy as np
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from scipy.spatial.transform import Rotation
from inference.robot_constants import GRIPPER_CLOSED_POSITION, GRIPPER_OPEN_POSITION


class RobotType(Enum):
    """Supported robot types."""

    WXAI = "wxai"
    STATIONARY_AI = "stationary_ai"
    MOBILE_AI = "mobile_ai"


# Robot configuration constants (WXAI defaults)
DEFAULT_ROBOT_PATH = "/World/wxai_robot"
END_EFFECTOR_LINK_NAME = "link_6"  # Common to all robot types

# Number of arm joints (excludes gripper)
NUM_ARM_JOINTS_WXAI = 6

# End effector offset from link_6 in meters [x, y, z]
EE_OFFSET = np.array([0.1055, 0.0, 0.0])

# Inverse kinematics parameters
DEFAULT_IK_SCALE = 0.5  # Scaling factor for joint velocity commands
DEFAULT_IK_DAMPING = 0.03  # Damping for singularity robustness
DEFAULT_MAX_JOINT_DELTA = 0.02  # Conservative per-command bound [rad]
DEFAULT_MAX_POSITION_ERROR = 0.05  # Maximum translational error fed to DLS [m]
DEFAULT_MAX_ORIENTATION_ERROR = 0.25  # Maximum rotational error fed to DLS [rad]

# Gripper position limits in meters


class UnsafeRobotStateError(RuntimeError):
    """Raised before an invalid state can be sent back into PhysX."""


def _normalize_quaternions(quaternions: np.ndarray, label: str) -> np.ndarray:
    """Return finite unit quaternions in Isaac's [w, x, y, z] order."""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"{label} must end in four quaternion values, got {values.shape}")
    if not np.isfinite(values).all():
        raise UnsafeRobotStateError(f"{label} contains NaN or infinity")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise UnsafeRobotStateError(f"{label} contains a zero-norm quaternion")
    normalized = values / norms
    # q and -q are equivalent. A single hemisphere prevents sign flips from
    # appearing as full rotations in telemetry and downstream interpolation.
    signs = np.where(normalized[..., :1] < 0.0, -1.0, 1.0)
    return normalized * signs


def _quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product for equally shaped [w, x, y, z] NumPy batches."""
    w1, x1, y1, z1 = np.moveaxis(first, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(second, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _bounded_vectors(vectors: np.ndarray, maximum_norm: float) -> np.ndarray:
    """Scale vectors down to maximum_norm without changing direction."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, maximum_norm / np.maximum(norms, 1.0e-12))
    return vectors * scale


class TrossenAIController(Articulation):
    """Controller for Trossen AI robots with differential IK and gripper control.

    Supported robots:
    - WidowX AI: 8 DOFs (6 arm joints + 2 gripper fingers)
    - Stationary AI: 16 DOFs (6 + 6 left and right arm joints, 2 + 2 left and right gripper fingers)
    - Mobile AI: 26 DOFs (4 + 4 caster wheels + caster swivel, 1 + 1 left and right wheels joints, 6 + 6 left and right arm joints, 2 + 2 gripper fingers)
    """

    def __init__(
        self,
        robot_path: str = DEFAULT_ROBOT_PATH,
        robot_type: RobotType = RobotType.WXAI,
        end_effector_link: RigidPrim | None = None,
        ik_scale: float = DEFAULT_IK_SCALE,
        ik_damping: float = DEFAULT_IK_DAMPING,
        arm_dof_indices: list[int] | None = None,
        gripper_dof_index: int | None = None,
        default_dof_positions: list[float] | None = None,
        max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA,
        max_position_error: float = DEFAULT_MAX_POSITION_ERROR,
        max_orientation_error: float = DEFAULT_MAX_ORIENTATION_ERROR,
        telemetry_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        """Initialize Trossen AI robot controller.

        Args:
            robot_path: USD scene path for the robot that already exists in the scene.
            robot_type: Type of robot (WXAI, STATIONARY_AI, or MOBILE_AI). Default: RobotType.WXAI
            end_effector_link: End effector link. If None, defaults to 'link_6'.
            ik_scale: Scaling factor for IK joint velocity commands (0.0-1.0). Default: 0.5
            ik_damping: Damping factor for singularity robustness (0.0-0.1). Default: 0.03
            arm_dof_indices: DOF indices for the arm joints. Required.
            gripper_dof_index: DOF index for the gripper. Required.
            default_dof_positions: Default joint configuration for reset. Required.
        """
        super().__init__(robot_path)

        self.robot_type = robot_type

        if arm_dof_indices is not None:
            self.arm_dof_indices = arm_dof_indices
        else:
            raise ValueError("arm_dof_indices must be provided")

        if gripper_dof_index is not None:
            self.gripper_dof_index = gripper_dof_index
        else:
            raise ValueError("gripper_dof_index must be provided")

        if default_dof_positions is not None:
            self.default_dof_positions = default_dof_positions
        else:
            raise ValueError("default_dof_positions must be provided")

        if end_effector_link is not None:
            self.end_effector_link = end_effector_link
            link_name = end_effector_link.paths[0].split("/")[-1]
        else:
            self.end_effector_link = RigidPrim(f"{robot_path}/{END_EFFECTOR_LINK_NAME}")
            link_name = END_EFFECTOR_LINK_NAME

        self.end_effector_link_index = self.get_link_indices(link_name).list()[0]
        self.ee_offset = EE_OFFSET

        self.ik_scale = ik_scale
        self.ik_damping = ik_damping
        self.max_joint_delta = float(max_joint_delta)
        self.max_position_error = float(max_position_error)
        self.max_orientation_error = float(max_orientation_error)
        self.telemetry_sink = telemetry_sink
        self.command_index = 0
        self.last_telemetry: dict[str, Any] = {}
        self._arm_dof_limits: tuple[np.ndarray, np.ndarray] | None = None
        self._last_ik_metrics: dict[str, Any] = {}

        if self.max_joint_delta <= 0.0:
            raise ValueError("max_joint_delta must be positive")
        if self.max_position_error <= 0.0 or self.max_orientation_error <= 0.0:
            raise ValueError("IK error bounds must be positive")

        self.set_default_state(dof_positions=self.default_dof_positions)

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint velocity commands using damped least-squares IK.

        Args:
            jacobian_end_effector: End effector Jacobian matrix (batch_size, 6, num_joints).
            current_position: Current end effector position (batch_size, 3).
            current_orientation: Current orientation quaternion (batch_size, 4) [w, x, y, z].
            goal_position: Target position (batch_size, 3).
            goal_orientation: Target orientation quaternion (batch_size, 4). Uses current if None.

        Returns:
            Joint position deltas (batch_size, num_joints).
        """
        jacobian_end_effector = np.asarray(jacobian_end_effector, dtype=np.float64)
        current_position = np.asarray(current_position, dtype=np.float64)
        goal_position = np.asarray(goal_position, dtype=np.float64)
        if not np.isfinite(jacobian_end_effector).all():
            raise UnsafeRobotStateError("End-effector Jacobian contains NaN or infinity")
        if not np.isfinite(current_position).all() or not np.isfinite(goal_position).all():
            raise UnsafeRobotStateError("IK position contains NaN or infinity")

        current_orientation = _normalize_quaternions(
            current_orientation, "current wrist orientation"
        )
        goal_orientation = _normalize_quaternions(
            current_orientation if goal_orientation is None else goal_orientation,
            "goal wrist orientation",
        )

        current_conjugate = current_orientation.copy()
        current_conjugate[:, 1:] *= -1.0
        q_error = _quaternion_multiply_wxyz(goal_orientation, current_conjugate)
        q_error *= np.where(q_error[:, :1] < 0.0, -1.0, 1.0)

        position_error_raw = goal_position - current_position
        position_error = _bounded_vectors(position_error_raw, self.max_position_error)

        vector = q_error[:, 1:]
        vector_norm = np.linalg.norm(vector, axis=1, keepdims=True)
        angle = 2.0 * np.arctan2(vector_norm, np.clip(q_error[:, :1], 0.0, 1.0))
        axis = vector / np.maximum(vector_norm, 1.0e-12)
        orientation_error_raw = axis * angle
        orientation_error = _bounded_vectors(
            orientation_error_raw, self.max_orientation_error
        )

        error = np.expand_dims(
            np.concatenate([position_error, orientation_error], axis=-1), axis=2
        )

        transpose = np.swapaxes(jacobian_end_effector, 1, 2)
        damping_matrix = np.eye(jacobian_end_effector.shape[1]) * (self.ik_damping**2)
        system = jacobian_end_effector @ transpose + damping_matrix
        try:
            solved = np.linalg.solve(system, error)
        except np.linalg.LinAlgError as exc:
            raise UnsafeRobotStateError("DLS system is singular") from exc
        delta_raw = (self.ik_scale * transpose @ solved).squeeze(-1)
        if not np.isfinite(delta_raw).all():
            raise UnsafeRobotStateError("DLS produced a non-finite joint update")
        delta = np.clip(delta_raw, -self.max_joint_delta, self.max_joint_delta)

        try:
            singular_values = np.linalg.svd(jacobian_end_effector[0], compute_uv=False)
            min_singular = float(np.min(singular_values))
            max_singular = float(np.max(singular_values))
        except np.linalg.LinAlgError:
            min_singular = 0.0
            max_singular = 0.0
        self._last_ik_metrics = {
            "position_error_raw_m": position_error_raw[0].tolist(),
            "position_error_used_m": position_error[0].tolist(),
            "orientation_error_raw_rad": orientation_error_raw[0].tolist(),
            "orientation_error_used_rad": orientation_error[0].tolist(),
            "raw_max_joint_delta_rad": float(np.max(np.abs(delta_raw))),
            "applied_max_joint_delta_rad": float(np.max(np.abs(delta))),
            "joint_delta_clamped": bool(np.any(np.abs(delta_raw) > self.max_joint_delta)),
            "jacobian_min_singular_value": min_singular,
            "jacobian_condition": (
                float(max_singular / min_singular) if min_singular > 1.0e-12 else None
            ),
        }
        return delta

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current robot state.

        Returns:
            Tuple of (joint_positions, end_effector_position, end_effector_orientation).
            Shapes: (batch_size, num_dofs), (batch_size, 3), (batch_size, 4)
        """
        current_dof_positions = self.get_dof_positions().numpy()
        wrist_position, wrist_orientation = self.end_effector_link.get_world_poses()
        wrist_position = np.asarray(wrist_position.numpy(), dtype=np.float64)
        wrist_orientation = _normalize_quaternions(
            wrist_orientation.numpy(), "physics wrist orientation"
        )
        if not np.isfinite(current_dof_positions).all():
            raise UnsafeRobotStateError("Physics joint state contains NaN or infinity")
        if not np.isfinite(wrist_position).all() or np.any(np.abs(wrist_position) > 100.0):
            raise UnsafeRobotStateError(
                f"Physics wrist position is invalid: {wrist_position.tolist()}"
            )

        current_end_effector_position = self._transform_offset_to_world(
            wrist_position, wrist_orientation, self.ee_offset
        )
        current_end_effector_orientation = wrist_orientation

        return (
            current_dof_positions,
            current_end_effector_position,
            current_end_effector_orientation,
        )

    def _transform_offset_to_world(
        self, position: np.ndarray, orientation: np.ndarray, offset: np.ndarray
    ) -> np.ndarray:
        """Transform local offset to world coordinates.

        Args:
            position: Position in world frame (batch_size, 3).
            orientation: Orientation quaternion [w, x, y, z] (batch_size, 4).
            offset: Local offset vector [x, y, z].

        Returns:
            World position with offset applied (batch_size, 3).
        """
        quat_scipy = orientation[:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat_scipy)
        offset_world = rotation.apply(np.tile(offset, (position.shape[0], 1)))
        return position + offset_world

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """Command end effector to target Cartesian pose using differential IK.

        Args:
            position: Target position [x, y, z] in meters. Shape: (3,) or (1, 3).
            orientation: Target orientation quaternion [w, x, y, z]. Shape: (4,) or (1, 4).
        """

        (
            current_dof_positions,
            _,
            _,
        ) = self.get_current_state()

        position = np.asarray(position, dtype=np.float64).reshape(1, -1)
        orientation = _normalize_quaternions(
            np.asarray(orientation, dtype=np.float64).reshape(1, -1),
            "commanded end-effector orientation",
        )
        if position.shape != (1, 3) or not np.isfinite(position).all():
            raise ValueError(f"End-effector position must be one finite XYZ row, got {position}")
        if np.any(np.abs(position) > 10.0):
            raise UnsafeRobotStateError(f"End-effector target is outside the scene: {position}")

        goal_wrist_position = self._transform_ee_to_wrist_frame(position, orientation)

        jacobian_matrices = self.get_jacobian_matrices().numpy()

        # Mobile base robots use direct link indices; fixed-base robots need -1 offset
        link_idx = (
            self.end_effector_link_index
            if self.robot_type == RobotType.MOBILE_AI
            else self.end_effector_link_index - 1
        )
        jacobian_full = jacobian_matrices[:, link_idx, :, :]

        # Isaac Sim releases differ on whether the Jacobian includes virtual
        # base columns. Resolve the offset from the runtime shapes instead of
        # assuming either layout. This remains zero for the original Trossen
        # Isaac Sim 5.1 asset/controller combination.
        num_robot_dofs = current_dof_positions.shape[1]
        base_dof_offset = max(0, jacobian_full.shape[-1] - num_robot_dofs)

        if self.robot_type in (RobotType.WXAI, RobotType.STATIONARY_AI):
            jacobian_indices = [
                base_dof_offset + idx for idx in self.arm_dof_indices
            ]
            jacobian_end_effector = jacobian_full[:, :, jacobian_indices]
        elif self.robot_type == RobotType.MOBILE_AI:
            # Mobile base adds 6 floating base DOFs before arm joints in Jacobian
            jacobian_indices = [idx + 6 for idx in self.arm_dof_indices]
            jacobian_end_effector = jacobian_full[:, :, jacobian_indices]
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")

        wrist_position, wrist_orientation = self.end_effector_link.get_world_poses()
        wrist_position = np.asarray(wrist_position.numpy(), dtype=np.float64)
        wrist_orientation = _normalize_quaternions(
            wrist_orientation.numpy(), "physics wrist orientation"
        )

        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=wrist_position,
            current_orientation=wrist_orientation,
            goal_position=goal_wrist_position,
            goal_orientation=orientation,
        )

        dof_position_targets = (
            current_dof_positions[:, self.arm_dof_indices] + delta_dof_positions
        )
        if self._arm_dof_limits is None:
            lower, upper = self.get_dof_limits(dof_indices=self.arm_dof_indices)
            self._arm_dof_limits = (
                np.asarray(lower.numpy(), dtype=np.float64),
                np.asarray(upper.numpy(), dtype=np.float64),
            )
        lower, upper = self._arm_dof_limits
        finite_limits = np.isfinite(lower) & np.isfinite(upper)
        limited_targets = np.where(
            finite_limits,
            np.minimum(np.maximum(dof_position_targets, lower), upper),
            dof_position_targets,
        )
        if not np.isfinite(limited_targets).all():
            raise UnsafeRobotStateError("Joint limit clamp produced a non-finite target")
        self.set_dof_position_targets(
            limited_targets, dof_indices=self.arm_dof_indices
        )

        self.command_index += 1
        current_tcp_position = self._transform_offset_to_world(
            wrist_position, wrist_orientation, self.ee_offset
        )
        telemetry: dict[str, Any] = {
            "command_index": self.command_index,
            "current_tcp_position_m": current_tcp_position[0].tolist(),
            "current_wrist_quaternion_wxyz": wrist_orientation[0].tolist(),
            "target_tcp_position_m": position[0].tolist(),
            "target_quaternion_wxyz": orientation[0].tolist(),
            "current_arm_joints": current_dof_positions[0, self.arm_dof_indices].tolist(),
            "target_arm_joints": limited_targets[0].tolist(),
            "joint_limit_clamped": bool(np.any(limited_targets != dof_position_targets)),
            **self._last_ik_metrics,
        }
        self.last_telemetry = telemetry
        if self.telemetry_sink is not None:
            self.telemetry_sink(telemetry)

    def _transform_ee_to_wrist_frame(
        self, ee_position: np.ndarray, ee_orientation: np.ndarray
    ) -> np.ndarray:
        """Convert end effector goal to wrist (link_6) frame.

        Args:
            ee_position: End effector position (batch_size, 3).
            ee_orientation: End effector orientation quaternion [w, x, y, z] (batch_size, 4).

        Returns:
            Wrist position in world frame (batch_size, 3).
        """
        quat_scipy = ee_orientation[:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat_scipy)
        offset_world = rotation.apply(
            np.tile(self.ee_offset, (ee_position.shape[0], 1))
        )
        return ee_position - offset_world

    def open_gripper(self) -> None:
        """Open the gripper to maximum width (0.044 meters)."""
        self.set_gripper_position(GRIPPER_OPEN_POSITION)

    def close_gripper(self) -> None:
        """Close the gripper around object (0.022 meters)."""
        self.set_gripper_position(GRIPPER_CLOSED_POSITION)

    def set_gripper_position(self, position: float) -> None:
        """Set gripper opening width.

        Args:
            position: Gripper width in meters (0.022 = closed, 0.044 = open).
        """
        self.set_dof_position_targets(
            np.array([[position]]), dof_indices=[self.gripper_dof_index]
        )

    def reset_to_default_pose(self) -> None:
        """Reset robot to default configuration with all joints at zero and gripper open."""
        default_positions = np.array([self.default_dof_positions])
        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)
