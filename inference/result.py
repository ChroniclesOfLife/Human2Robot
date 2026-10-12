"""Physical placement gates and JSON-safe failure reporting."""

import math

import numpy as np


def sanitize_result(result: dict) -> dict:
    invalid = []

    def clean(value, path):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            invalid.append(path)
            return None
        if isinstance(value, dict):
            return {key: clean(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    safe = clean(result, "result")
    if invalid:
        safe.update(physical_success=False, previous_termination_reason=result.get("termination_reason"),
                    termination_reason="nonfinite_physics_result", nonfinite_fields=invalid)
    return safe


class PhysicalSuccessTracker:
    """Evaluate live object readback; policy done is deliberately not an input."""

    def __init__(self, initial_bread_pose, minimum_lift_m=0.03, maximum_place_distance_m=0.05,
                 minimum_release_seconds=0.5, settled_seconds=0.25, maximum_speed_m_s=0.02):
        self.initial = np.asarray(initial_bread_pose, dtype=np.float64)[:3, 3].copy()
        self.previous = self.initial.copy()
        self.minimum_lift_m = minimum_lift_m
        self.maximum_place_distance_m = maximum_place_distance_m
        self.minimum_release_seconds = minimum_release_seconds
        self.settled_seconds = settled_seconds
        self.maximum_speed_m_s = maximum_speed_m_s
        self.lift_m = 0.0
        self.saw_closed = False
        self.release_seconds = 0.0
        self.stable_seconds = 0.0
        self.metrics = {"physical_success": False}

    def observe(self, bread_pose, plate_pose, *, gripper_closed, attached, dt, plate_points_world=None):
        bread = np.asarray(bread_pose, dtype=np.float64)[:3, 3]
        plate = np.asarray(plate_pose, dtype=np.float64)[:3, 3]
        if dt <= 0 or not np.isfinite(bread).all() or not np.isfinite(plate).all():
            raise ValueError("Physical evaluation requires finite measured poses and positive dt")
        self.lift_m = max(self.lift_m, float(bread[2] - self.initial[2]))
        speed = float(np.linalg.norm(bread - self.previous) / dt)
        self.previous = bread.copy()
        closed = float(gripper_closed) > 0.5
        self.saw_closed |= closed
        released = self.saw_closed and not closed and not attached
        self.release_seconds = self.release_seconds + dt if released else 0.0
        distance = float(np.linalg.norm(bread - plate))
        height = float(bread[2] - plate[2])
        inside_plate = True
        if plate_points_world is not None:
            points = np.asarray(plate_points_world, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Plate support requires at least three finite measured keypoints")
            # Convex footprint of the calibrated plate boundary in world XY.
            from scipy.spatial import ConvexHull, QhullError
            try:
                hull = ConvexHull(points[:, :2])
                inside_plate = bool(np.all(hull.equations[:, :2] @ bread[:2] + hull.equations[:, 2] <= 0.005))
            except QhullError:
                inside_plate = False
        support_geometry = inside_plate and -0.005 <= height <= 0.06
        settled = (released and support_geometry and distance <= self.maximum_place_distance_m
                   and speed <= self.maximum_speed_m_s)
        self.stable_seconds = self.stable_seconds + dt if settled else 0.0
        success = (self.lift_m >= self.minimum_lift_m and released
                   and self.release_seconds >= self.minimum_release_seconds
                   and self.stable_seconds >= self.settled_seconds)
        self.metrics = {"physical_success": bool(success), "bread_lift_m": self.lift_m,
                        "bread_to_plate_m": distance, "bread_speed_m_s": speed,
                        "release_seconds": self.release_seconds, "settled_seconds": self.stable_seconds,
                        "minimum_lift_m": self.minimum_lift_m,
                        "maximum_place_distance_m": self.maximum_place_distance_m,
                        "minimum_release_seconds": self.minimum_release_seconds,
                        "minimum_settled_seconds": self.settled_seconds,
                        "maximum_settled_speed_m_s": self.maximum_speed_m_s}
        self.metrics.update(plate_height_offset_m=height, inside_plate_footprint=inside_plate,
                            support_geometry_pass=bool(support_geometry),
                            support_gate="measured_plate_footprint_and_height" if plate_points_world is not None else "center_distance_and_height")
        return self.metrics.copy()
