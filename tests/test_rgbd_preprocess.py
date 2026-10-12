"""CPU checks using synthetic geometry, never model downloads or demo generation."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess.geometry import PoseSmoother, hand_from_keypoints, lift_pixels, object_from_mask
from preprocess.models import Inpainter, MediaPipeHandTracker
from preprocess.prepare_rgbd import load_depth, main
import preprocess.prepare_rgbd as pipeline


def hand_pixels():
    points = np.tile([40.0, 45.0], (21, 1))
    points[0] = [40, 65]
    points[4] = [35, 30]
    points[8] = [45, 30]
    points[5] = [48, 47]
    points[9] = [40, 45]
    points[17] = [32, 50]
    return points


def test_lift_uses_metric_depth_and_full_intrinsics():
    k = np.array([[100, 10, 20], [0, 200, 30], [0, 0, 1]])
    depth = np.full((80, 80), 2.0)
    xyz, valid = lift_pixels(np.array([[31, 50], [20, 30]]), depth, k, radius=0)
    np.testing.assert_allclose(xyz, [[0.2, 0.2, 2], [0, 0, 2]])
    assert valid.tolist() == [True, True]


def test_invalid_depth_and_outside_pixels_do_not_produce_poses():
    depth = np.ones((4, 4))
    depth[1, 1] = np.nan
    points, valid = lift_pixels(np.array([[1, 1], [-1, 2], [4, 2]]), depth, np.eye(3), radius=0)
    assert not valid.any()
    assert np.isnan(points).all()


def test_standard_thumb_and_index_tips_define_the_grasp_frame():
    k = np.array([[400, 0, 40], [0, 400, 40], [0, 0, 1]])
    hand = hand_from_keypoints(hand_pixels(), np.full((80, 80), 0.8), k)
    np.testing.assert_allclose(hand["T_hand_to_world"][:3, 3], [0, -0.02, 0.8], atol=1e-9)
    assert hand["grasp"] == 1.0
    assert hand["pinch_distance_m"] == pytest.approx(0.02)
    rotation = hand["T_hand_to_world"][:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_grasp_hysteresis_depends_on_measured_tip_distance():
    k = np.array([[400, 0, 40], [0, 400, 40], [0, 0, 1]])
    points = hand_pixels()
    points[4, 0], points[8, 0] = 30, 50  # 40 mm at 0.8 m
    depth = np.full((80, 80), 0.8)
    assert hand_from_keypoints(points, depth, k, previous_grasp=0.0)["grasp"] == 0.0
    assert hand_from_keypoints(points, depth, k, previous_grasp=1.0)["grasp"] == 1.0


def test_missing_or_degenerate_landmarks_are_rejected():
    k = np.array([[400, 0, 40], [0, 400, 40], [0, 0, 1]])
    assert hand_from_keypoints(hand_pixels(), np.zeros((80, 80)), k) is None
    assert hand_from_keypoints(np.tile([40, 40], (21, 1)), np.ones((80, 80)), k) is None
    with pytest.raises(ValueError):
        hand_from_keypoints(np.zeros((20, 2)), np.ones((80, 80)), k)


def test_object_pca_points_remain_consistent_with_the_pose():
    k = np.array([[200, 0, 40], [0, 200, 40], [0, 0, 1]])
    mask = np.zeros((80, 80), bool)
    mask[25:35, 20:60] = True
    obj = object_from_mask(mask, np.full((80, 80), 1.2), k)
    pose = obj["T_obj_to_world"]
    np.testing.assert_allclose(pose[:3, 3], [-0.003, -0.063, 1.2], atol=1e-8)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(obj["local"] @ pose[:3, :3].T + pose[:3, 3], obj["world"], atol=1e-8)
    assert object_from_mask(mask, np.zeros((80, 80)), k) is None


def test_smoothing_preserves_rotations_and_resets_after_tracking_loss():
    smoother = PoseSmoother(alpha=0.5)
    first, second = np.eye(4), np.eye(4)
    first[2, 3] = second[2, 3] = 1
    second[0, 3] = 0.2
    smoother.update("right", first)
    middle = smoother.update("right", second)
    assert middle[0, 3] == pytest.approx(0.1)
    np.testing.assert_allclose(middle[:3, :3].T @ middle[:3, :3], np.eye(3))
    smoother.reset("right")
    assert smoother.update("right", second)[0, 3] == pytest.approx(0.2)


def write_recording(root: Path, *, labels=True):
    k = [[400, 0, 40], [0, 400, 40], [0, 0, 1]]
    root.mkdir()
    (root / "intrinsics.json").write_text(json.dumps({"k": k, "fps": 30}))
    for folder in ("rgb", "depth", "hands", "masks"):
        (root / folder).mkdir()
    for i in range(3):
        stem = f"{i:05d}"
        Image.fromarray(np.full((80, 80, 3), 100, np.uint8)).save(root / "rgb" / f"{stem}.png")
        np.save(root / "depth" / f"{stem}.npy", np.full((80, 80), 0.8, np.float32))
        (root / "hands" / f"{stem}.json").write_text(json.dumps({"right": {"keypoints": hand_pixels().tolist()}}))
        mask_dir = root / "masks" / stem
        mask_dir.mkdir()
        for name, bounds in {"obj1": (10, 20, 15, 35), "obj2": (50, 65, 15, 65), "arm": (65, 75, 35, 45)}.items():
            mask = np.zeros((80, 80), np.uint8)
            y0, y1, x0, x1 = bounds
            mask[y0:y1, x0:x1] = 255
            Image.fromarray(mask).save(mask_dir / f"{name}.png")
    if labels:
        (root / "labels.json").write_text(json.dumps({"00000": False, "00001": False, "00002": True}))


def offline_args(recording, output):
    return ["--input", str(recording), "--output", str(output), "--hand-backend", "json", "--object-backend", "masks", "--inpaint", "none"]


def test_offline_export_produces_real_metric_targets_and_visual_observation(tmp_path):
    recording, output = tmp_path / "recording", tmp_path / "session"
    write_recording(recording)
    assert main(offline_args(recording, output)) == 0
    frame = output / "preprocess/all_data/00002"
    data = json.loads((frame / "training_data.json").read_text())
    assert data["metadata"]["is_finished"] == 1.0
    assert data["metadata"]["anchor_key"] == "obj1"
    np.testing.assert_allclose(data["metadata"]["c2w"], np.eye(4))
    np.testing.assert_allclose(data["entities"]["hands"]["right"]["T_hand_to_world"][2][3], 0.8)
    assert set(data["entities"]["objects"]) == {"obj1", "obj2"}
    assert data["metadata"]["preprocessing"]["inpaint"] == "none"
    assert data["obs"]["observation_path"] == "observation.png"
    observation = np.asarray(Image.open(frame / "observation.png"))
    assert np.any(observation != 100)  # geometric gripper/keypoint overlay actually rendered
    manifest = json.loads((output / "preprocess/manifest.json").read_text())
    assert manifest["accepted_frames"] == 3
    with pytest.raises(FileExistsError):
        main(offline_args(recording, output))


def test_export_requires_explicit_completion_labels(tmp_path):
    recording = tmp_path / "recording"
    write_recording(recording, labels=False)
    with pytest.raises(FileNotFoundError):
        main(offline_args(recording, tmp_path / "session"))


def test_tracking_loss_leaves_a_gap_instead_of_fabricating_targets(tmp_path):
    recording, output = tmp_path / "recording", tmp_path / "session"
    write_recording(recording)
    np.save(recording / "depth/00001.npy", np.zeros((80, 80), np.float32))
    assert main(offline_args(recording, output)) == 0
    assert not (output / "preprocess/all_data/00001/training_data.json").exists()
    manifest = json.loads((output / "preprocess/manifest.json").read_text())
    assert manifest["accepted_frames"] == 2
    assert manifest["rejected_frames"][0]["source_frame"] == "00001"


def test_png_depth_requires_an_explicit_metric_scale(tmp_path):
    (tmp_path / "depth").mkdir()
    Image.fromarray(np.full((5, 5), 800, np.uint16)).save(tmp_path / "depth/00000.png")
    with pytest.raises(ValueError, match="requires"):
        load_depth(tmp_path, "00000")
    np.testing.assert_allclose(load_depth(tmp_path, "00000", scale=0.001), 0.8)


def test_missing_weights_are_reported_before_optional_imports():
    with pytest.raises(FileNotFoundError):
        Inpainter("lama")
    with pytest.raises(FileNotFoundError):
        MediaPipeHandTracker(None)


def test_config_controls_geometry_and_cli_can_override_it(tmp_path):
    recording, output = tmp_path / "recording", tmp_path / "session"
    write_recording(recording)
    config = tmp_path / "settings.yaml"
    config.write_text("hand_origin: palm\nsmoothing_alpha: 1.0\n")
    args = offline_args(recording, output) + ["--config", str(config), "--hand-origin", "pinch"]
    assert main(args) == 0
    data = json.loads((output / "preprocess/all_data/00000/training_data.json").read_text())
    assert data["metadata"]["preprocessing"]["smoothing_alpha"] == 1.0
    assert data["entities"]["hands"]["right"]["pose_definition"] == "pinch"


def test_numerically_named_frames_follow_capture_order(tmp_path):
    recording, output = tmp_path / "recording", tmp_path / "session"
    write_recording(recording)
    names = {"00000": "1", "00001": "2", "00002": "10"}
    for folder, extension in (("rgb", "png"), ("depth", "npy"), ("hands", "json")):
        for before, after in names.items():
            (recording / folder / f"{before}.{extension}").rename(recording / folder / f"{after}.{extension}")
    for before, after in names.items():
        (recording / "masks" / before).rename(recording / "masks" / after)
    (recording / "labels.json").write_text(json.dumps({"1": False, "2": False, "10": True}))
    assert main(offline_args(recording, output)) == 0
    data = json.loads((output / "preprocess/all_data/00002/training_data.json").read_text())
    assert data["metadata"]["source_frame"] == "10"
    assert data["metadata"]["is_finished"] == 1.0


def test_anchor_uses_first_accepted_frame_after_missing_arm_mask(tmp_path, monkeypatch):
    recording, output = tmp_path / "recording", tmp_path / "session"
    write_recording(recording)
    Image.fromarray(np.zeros((80, 80), np.uint8)).save(recording / "masks/00000/arm.png")
    np.save(recording / "depth/00001.npy", np.full((80, 80), 1.0, np.float32))
    # Only substitute the unavailable external ONNX inference; retain real input,
    # segmentation, hand/object geometry, rejection logic and output persistence.
    class IdentityModel:
        def __init__(self, *args, **kwargs):
            pass

        def inpaint(self, image, mask):
            return image.copy()

    monkeypatch.setattr(pipeline, "Inpainter", IdentityModel)
    args = offline_args(recording, output)
    args[-1] = "lama"
    assert main(args) == 0
    data = json.loads((output / "preprocess/all_data/00001/training_data.json").read_text())
    assert data["metadata"]["world_transforms"]["virtual_static_anchor"][2][3] == pytest.approx(1.0)
    manifest = json.loads((output / "preprocess/manifest.json").read_text())
    assert manifest["rejected_frames"][0]["reason"] == "missing_arm_mask"
