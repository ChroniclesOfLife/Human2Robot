"""Robot adapter contracts: meters, optical camera poses, 0=open / 1=closed.

A new physical robot needs calibrated camera/base and hand/EE transforms,
measured FK, a bounded Cartesian servo/IK implementation, gripper mapping,
and a perception pipeline matching the training image and ICT conventions.
The WXAI Isaac adapter is the packaged example; other robots need adapters.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray  # BGR uint8
    depth_m: np.ndarray  # metric optical depth, NaN/0 invalid
    K: np.ndarray


@dataclass
class ObjectState:
    T_in_cam: np.ndarray
    kpts_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


class Camera(Protocol):
    def get_frame(self) -> Frame: ...
    def close(self) -> None: ...


class RobotArm(Protocol):
    T_base_in_cam: np.ndarray

    def get_T_ee_in_cam(self) -> np.ndarray: ...
    def move_ee_in_cam(self, T_ee_in_cam: np.ndarray, duration: float, blocking: bool = False) -> bool: ...
    def get_gripper(self) -> float: ...
    def set_gripper(self, value: float, blocking: bool = False) -> None: ...
    def go_home(self, blocking: bool = True) -> None: ...
    def close(self) -> None: ...


class Perception(Protocol):
    def estimate_objects(self, frames: list[Frame]) -> dict[str, ObjectState]: ...
    def make_clean_image(self, frame: Frame, hands_in_cam: dict[str, np.ndarray], grippers: dict[str, float]) -> np.ndarray: ...
