"""Metric geometry in the fixed camera's world frame (x right, y down, z forward)."""

import numpy as np


def lift_pixels(pixels, depth_m, k, *, radius=2, min_depth=0.1, max_depth=5.0):
    """Back-project pixel coordinates using valid median depth in a local patch.

    Invalid samples remain NaN and are accompanied by a false validity bit; no
    depth is inferred from physical hand size or copied from a previous frame.
    """
    pixels = np.asarray(pixels, dtype=float)
    depth_m, k = np.asarray(depth_m), np.asarray(k, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth_m.ndim != 2:
        raise ValueError("Expected Nx2 pixels and aligned HxW metric depth")
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("Camera intrinsics must be a finite, invertible 3x3 matrix")
    if radius < 0 or not 0 < min_depth < max_depth:
        raise ValueError("Invalid depth sampling limits")
    h, w = depth_m.shape
    xyz = np.full((len(pixels), 3), np.nan)
    valid = np.zeros(len(pixels), bool)
    k_inv = np.linalg.inv(k)
    for i, (u, v) in enumerate(pixels):
        if not np.isfinite([u, v]).all() or not (0 <= u < w and 0 <= v < h):
            continue
        x, y = int(round(u)), int(round(v))
        x, y = min(x, w - 1), min(y, h - 1)
        patch = depth_m[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        values = patch[np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)]
        if not len(values):
            continue
        ray = k_inv @ np.array([u, v, 1.0])
        xyz[i] = ray / ray[2] * float(np.median(values))
        valid[i] = np.isfinite(xyz[i]).all()
    return xyz, valid


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(norm) and norm > 1e-6 else None


def hand_from_keypoints(pixels, depth_m, k, *, previous_grasp=0.0, close_m=0.035,
                        open_m=0.05, origin="pinch", radius=2, min_depth=0.1,
                        max_depth=5.0):
    """Build a right-handed grasp frame from standard 21 hand landmarks.

    Wrist=0, thumb tip=4, index tip=8, index/middle/pinky MCP=5/9/17.
    Local x spans thumb to index, local y points from wrist toward the grasp.
    An almost closed pinch uses the palm's across-finger vector for x.
    """
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (21, 2):
        raise ValueError("Hand keypoints must have shape (21, 2) in RGB pixels")
    if not 0 < close_m < open_m or origin not in {"pinch", "palm"}:
        raise ValueError("Invalid grasp thresholds or hand origin")
    points, valid = lift_pixels(pixels, depth_m, k, radius=radius, min_depth=min_depth, max_depth=max_depth)
    required = [0, 4, 8, 5, 9, 17]
    if not valid[required].all() or valid.sum() < 12:
        return None
    tip_span = points[8] - points[4]
    pinch = (points[4] + points[8]) * 0.5
    palm = points[[0, 5, 9, 17]].mean(axis=0)
    centre = pinch if origin == "pinch" else palm
    y = _unit(points[9] - points[0])
    if y is None:
        return None
    across = tip_span if np.linalg.norm(tip_span) > 0.005 else points[5] - points[17]
    x = _unit(across - y * np.dot(across, y))
    if x is None:
        return None
    z = _unit(np.cross(x, y))
    y = np.cross(z, x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, y, z))
    transform[:3, 3] = centre
    distance = float(np.linalg.norm(tip_span))
    threshold = open_m if previous_grasp > 0.5 else close_m
    return {"T_hand_to_world": transform, "grasp": float(distance < threshold),
            "pinch_distance_m": distance, "keypoints_world": points,
            "keypoints_valid": valid, "pose_definition": origin}


def object_from_mask(mask, depth_m, k, *, previous_rotation=None, min_points=30,
                     max_points=4000, radius=0, min_depth=0.1, max_depth=5.0):
    """Estimate a visible-surface centroid and PCA axes, not a full CAD pose.

    A symmetric object has an ambiguous in-plane orientation. In that case a
    camera-right tangent axis (or the prior measured axis) fixes the convention;
    the ambiguity is recorded for consumers. Masked depth outliers are trimmed.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != np.asarray(depth_m).shape:
        raise ValueError("Object mask and depth must be aligned")
    y, x = np.nonzero(mask)
    if len(x) < min_points:
        return None
    indices = np.linspace(0, len(x) - 1, min(len(x), max_points), dtype=int)
    points, valid = lift_pixels(np.column_stack((x[indices], y[indices])), depth_m, k,
                                radius=radius, min_depth=min_depth, max_depth=max_depth)
    points = points[valid]
    if len(points) < min_points:
        return None
    median_z = np.median(points[:, 2])
    mad_z = np.median(np.abs(points[:, 2] - median_z))
    points = points[np.abs(points[:, 2] - median_z) <= max(0.05, 3 * 1.4826 * mad_z)]
    if len(points) < min_points:
        return None
    centre = points.mean(axis=0)
    values, axes = np.linalg.eigh(np.cov((points - centre).T))
    if values[1] < 1e-9:
        return None
    z = axes[:, 0]
    if np.dot(z, centre) < 0:
        z = -z
    ambiguous = values[2] / values[1] < 1.15
    reference = np.asarray(previous_rotation)[:, 0] if previous_rotation is not None else np.array([1., 0., 0.])
    x = axes[:, 2]
    if ambiguous:
        tangent = reference - z * np.dot(reference, z)
        if _unit(tangent) is not None:
            x = _unit(tangent)
    if np.dot(x, reference) < 0:
        x = -x
    y = _unit(np.cross(z, x))
    x = np.cross(y, z)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, y, z))
    transform[:3, 3] = centre
    indices = np.linspace(0, len(points) - 1, min(64, len(points)), dtype=int)
    world = points[indices]
    local = (world - centre) @ transform[:3, :3]
    return {"T_obj_to_world": transform, "world": world, "local": local,
            "orientation_ambiguous": bool(ambiguous), "valid_depth_points": len(points),
            "pose_definition": "visible_surface_centroid_pca"}


class PoseSmoother:
    """Causal EMA for depth noise; project averaged rotation onto SO(3).

    alpha=1 disables smoothing. Tracking gaps must call reset, so a measurement
    is never extrapolated across occlusion. This filter introduces temporal lag.
    """

    def __init__(self, alpha=0.5):
        if not 0 < alpha <= 1:
            raise ValueError("Smoothing alpha must lie in (0, 1]")
        self.alpha, self.previous = alpha, {}

    def reset(self, key):
        self.previous.pop(key, None)

    def update(self, key, transform):
        current = np.asarray(transform, dtype=float).copy()
        if current.shape != (4, 4) or not np.isfinite(current).all():
            raise ValueError("A measured pose must be a finite 4x4 transform")
        if key in self.previous:
            previous = self.previous[key]
            a = self.alpha
            current[:3, 3] = a * current[:3, 3] + (1 - a) * previous[:3, 3]
            u, _, vt = np.linalg.svd(a * current[:3, :3] + (1 - a) * previous[:3, :3])
            correction = np.diag([1., 1., np.linalg.det(u @ vt)])
            current[:3, :3] = u @ correction @ vt
        self.previous[key] = current.copy()
        return current
