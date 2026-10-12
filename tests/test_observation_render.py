"""CPU checks for the checkpoint's virtual-gripper observation convention."""

from types import SimpleNamespace

import numpy as np
import pytest

from preprocess.visualize import render_observation


def inputs(grasp=0.0):
    pose = np.eye(4)
    pose[2, 3] = 1.0
    k = np.array([[1000., 0, 320], [0, 1000., 240], [0, 0, 1]])
    return np.zeros((480, 640, 3), np.uint8), k, {"right": {"T_hand_to_world": pose, "grasp": grasp}}


@pytest.mark.parametrize("grasp,tip_x,root,finger,tip", [
    (0.49, 230, [255, 0, 0], [255, 160, 0], [255, 255, 0]),
    (0.51, 295, [255, 0, 255], [255, 255, 255], [255, 255, 255]),
])
def test_checkpoint_gripper_dimensions_and_binary_palette(grasp, tip_x, root, finger, tip):
    rgb, k, hands = inputs(grasp)
    image = render_observation(rgb, hands, {}, k)
    np.testing.assert_array_equal(image[80, 323], root)  # root y=-0.16 metres
    np.testing.assert_array_equal(image[160, tip_x + 3], finger)  # bases y=-0.08
    np.testing.assert_array_equal(image[240, tip_x + 3], tip)  # pinch is the fingertip midpoint
    np.testing.assert_array_equal(image[240, tip_x], [255, 255, 255])
    np.testing.assert_array_equal(image[240, 320], [0, 0, 0])  # no marker at pose origin


def test_four_segments_have_six_discrete_points_and_warm_gradient():
    rgb, k, hands = inputs()
    image = render_observation(rgb, hands, {}, k)
    np.testing.assert_array_equal(image[128, 269], [255, 36, 0])  # root-to-left-base 4th sample
    np.testing.assert_array_equal(image[128, 377], [255, 36, 0])  # symmetric right segment
    np.testing.assert_array_equal(image[208, 233], [255, 217, 0])  # left finger 4th sample
    np.testing.assert_array_equal(image[208, 413], [255, 217, 0])  # right finger
    np.testing.assert_array_equal(image[184, 230], [0, 0, 0])  # separated dots, not solid lines


def test_object_markers_use_cyan_green_radius_five_and_white_core():
    rgb, k, _ = inputs()
    objects = {"obj1": {"world": [[-0.15, 0.05, 1]]}, "obj2": {"world": [[0.15, 0.05, 1]]}}
    image = render_observation(rgb, {}, objects, k)
    np.testing.assert_array_equal(image[290, 174], [0, 255, 255])
    np.testing.assert_array_equal(image[290, 474], [0, 255, 0])
    np.testing.assert_array_equal(image[290, 170], [255, 255, 255])
    np.testing.assert_array_equal(image[290, 176], [0, 0, 0])


def test_objects_are_drawn_after_hand_points_and_invalid_depth_is_ignored():
    rgb, k, hands = inputs()
    objects = {"obj1": {"world": [[0, -0.16, 1], [0, 0, -1], [np.nan, 0, 1]]}}
    image = render_observation(rgb, hands, objects, k)
    np.testing.assert_array_equal(image[80, 323], [0, 255, 255])
    np.testing.assert_array_equal(image[240, 320], [0, 0, 0])


def test_marker_radii_scale_with_reference_image_resolution():
    rgb, k, hands = inputs()
    rgb, k = rgb[::2, ::2], k.copy()
    k[:2] /= 2
    image = render_observation(rgb, hands, {}, k)
    np.testing.assert_array_equal(image[120, 117], [255, 255, 0])
    np.testing.assert_array_equal(image[120, 118], [0, 0, 0])


def test_simulator_bgr_adapter_matches_rgb_preprocessing_exactly():
    from inference.simulator_perception import draw_training_observation

    rgb, k, hands = inputs(0.8)
    rgb[:] = [19, 37, 71]
    object_pose = np.eye(4)
    object_pose[:3, 3] = [0.15, 0.05, 1]
    local = np.array([[0, 0, 0], [0.02, 0, 0]])
    objects = {"obj2": SimpleNamespace(T_in_cam=object_pose, kpts_local=local)}
    expected = render_observation(rgb, hands, {"obj2": {"world": local + [0.15, 0.05, 1]}}, k)
    result = draw_training_observation(rgb[:, :, ::-1], k,
                                       {"right": hands["right"]["T_hand_to_world"]}, {"right": 0.8}, objects)
    np.testing.assert_array_equal(result[:, :, ::-1], expected)
    np.testing.assert_array_equal(result[0, 0], [71, 37, 19])
