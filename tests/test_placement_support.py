"""Placement diagnostics must reject proximity without plausible plate support."""

import numpy as np

from inference.result import PhysicalSuccessTracker


def evaluate_placement(offset, points):
    initial, plate, bread = np.eye(4), np.eye(4), np.eye(4)
    initial[2, 3] = plate[2, 3] = 0.9
    tracker = PhysicalSuccessTracker(initial)
    bread[2, 3] = 1.0
    tracker.observe(bread, plate, gripper_closed=1, attached=True, dt=0.1, plate_points_world=points)
    bread[:3, 3] = plate[:3, 3] + offset
    for _ in range(12):
        metrics = tracker.observe(bread, plate, gripper_closed=0, attached=False, dt=0.1, plate_points_world=points)
    return metrics


def test_stable_bread_below_plate_never_passes():
    points = np.array([[-.08, -.08, .9], [.08, -.08, .9], [.08, .08, .9], [-.08, .08, .9]])
    assert not evaluate_placement([0, 0, -.10], points)["physical_success"]
    assert not evaluate_placement([0, 0, -.02], points)["support_geometry_pass"]


def test_inside_footprint_and_plausible_height_can_pass_after_release():
    points = np.array([[-.08, -.08, .9], [.08, -.08, .9], [.08, .08, .9], [-.08, .08, .9]])
    metrics = evaluate_placement([0, 0, .032], points)
    assert metrics["physical_success"]
    assert metrics["support_geometry_pass"]


def test_nearby_bread_outside_narrow_plate_cannot_pass():
    points = np.array([[-.01, -.08, .9], [.01, -.08, .9], [.01, .08, .9], [-.01, .08, .9]])
    metrics = evaluate_placement([.03, 0, .02], points)
    assert not metrics["physical_success"]
    assert not metrics["inside_plate_footprint"]
