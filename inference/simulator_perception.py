"""BGR simulator frontend for the shared checkpoint observation renderer."""

import numpy as np

from preprocess.visualize import render_observation


def draw_training_observation(background, K, hands, grippers, objects):
    """Return BGR observations; hand poses already locate the fingertip midpoint.

    Geometry, binary grasp palette, dot radii and object colours match the
    RGB preprocessing renderer. Object points are transformed to the camera
    frame; no further translation or hand-axis correction is applied here.
    """
    hand_entries = {side: {"T_hand_to_world": transform,
                           "grasp": float(np.clip(grippers.get(side, 0), 0, 1))}
                    for side, transform in hands.items()}
    keypoints = {}
    for key, obj in objects.items():
        local = np.asarray(obj.kpts_local, dtype=float).reshape(-1, 3)
        pose = np.asarray(obj.T_in_cam, dtype=float)
        keypoints[key] = {"world": local @ pose[:3, :3].T + pose[:3, 3]}
    rgb = np.ascontiguousarray(np.asarray(background)[:, :, ::-1])
    rendered = render_observation(rgb, hand_entries, keypoints, K)
    return np.ascontiguousarray(rendered[:, :, ::-1])
