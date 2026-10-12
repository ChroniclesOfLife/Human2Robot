"""Rate-limited receding-horizon execution of neural EE targets."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from inference.interfaces import RobotArm


class TrajectoryController:
    def __init__(self, arms: dict[str, RobotArm], cfg: dict):
        self.arms = arms
        self.alpha_pos = cfg.get("alpha_pos", 0.3)
        self.alpha_rot = cfg.get("alpha_rot", 0.3)
        self.max_pos_step = cfg.get("max_pos_step", 0.05)
        self.max_rot_step = cfg.get("max_rot_step", 0.25)
        self.grasp_threshold = cfg.get("grasp_threshold", 0.5)
        self.reset()

    def reset(self):
        self._last_cmd = {side: arm.get_T_ee_in_cam().copy() for side, arm in self.arms.items()}

    def _smooth(self, side, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("Neural trajectory contains an invalid EE pose")
        previous = self._last_cmd[side]
        delta = (1 - self.alpha_pos) * (target[:3, 3] - previous[:3, 3])
        delta *= min(1.0, self.max_pos_step / max(float(np.linalg.norm(delta)), 1e-12))
        angle = (Rotation.from_matrix(target[:3, :3]) * Rotation.from_matrix(previous[:3, :3]).inv()).magnitude()
        fraction = min(1 - self.alpha_rot, self.max_rot_step / max(angle, 1e-12))
        output = np.eye(4)
        output[:3, 3] = previous[:3, 3] + delta
        output[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([previous[:3, :3], target[:3, :3]])))(fraction).as_matrix()
        self._last_cmd[side] = output
        return output

    def execute_chunk(self, traj_per_arm, grasp_per_arm, dt, n_steps):
        for index in range(n_steps):
            for side, arm in self.arms.items():
                if side not in traj_per_arm or index >= len(traj_per_arm[side]):
                    continue
                target = self._smooth(side, traj_per_arm[side][index])
                grasp = float(np.asarray(grasp_per_arm[side][index]).reshape(-1)[0])
                if not np.isfinite(grasp):
                    raise ValueError("Neural policy produced a non-finite grasp value")
                arm.set_gripper(float(grasp > self.grasp_threshold), blocking=False)
                # In the simulator, blocking advances physics while servoing each tick.
                # A hardware adapter must implement the same duration contract.
                if not arm.move_ee_in_cam(target, duration=dt, blocking=True):
                    self._last_cmd[side] = arm.get_T_ee_in_cam().copy()

