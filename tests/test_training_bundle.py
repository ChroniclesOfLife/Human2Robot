"""Portable frame loading and bounded training regressions."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def _pose(x=0.0, z=1.0):
    pose = np.eye(4)
    pose[0, 3], pose[2, 3] = x, z
    return pose.tolist()


def _session(root, name):
    session = root / "serve_bread" / "demos" / name
    for index in (101, 102, 103):
        frame = session / "preprocess" / "all_data" / f"{index:05d}"
        frame.mkdir(parents=True)
        camera = _pose(10.0, 0.0)
        data = {
            "metadata": {
                "idx": index, "h": 16, "w": 16,
                "world_transforms": {"cam0": camera, "virtual_static_anchor": _pose(0, 0)},
                "c2w": camera, "k": [[10, 0, 8], [0, 10, 8], [0, 0, 1]],
                "anchor_key": "obj1", "is_finished": index == 103,
            },
            "obs": {"rgb_path": "/unavailable/old/image.png", "objects_kpts": {}},
            "entities": {
                "hands": {"right": {"T_hand_to_world": _pose(index - 89), "grasp": 1}},
                "objects": {"obj1": {"T_obj_to_world": _pose(11)}},
            },
        }
        (frame / "training_data.json").write_text(json.dumps(data))
        cv2.imwrite(str(frame / "observation.png"), np.full((16, 16, 3), [40, 100, 200], dtype=np.uint8))
    return session


@pytest.mark.parametrize("original_resolution", [False, True])
def test_frame_images_follow_json_location_and_preserve_camera_coordinates(tmp_path, monkeypatch, original_resolution):
    from training.FlowMatchingDataloader import FlowMatchingDataloader, SessionSpec

    session = _session(tmp_path / "data", "session_001")
    dataset = FlowMatchingDataloader(
        [SessionSpec(str(session))], image_size=(16, 16), pred_horizon=2,
        single_hand=True, frame_mode="camera_frame", hand_tracking_method="keypoints",
        img_name="observation.png", enable_augmentation=False,
        use_legacy_image_loading=original_resolution,
    )
    monkeypatch.chdir(tmp_path)
    sample = dataset[0]
    assert sample["meta_t"].item() == 101
    torch.testing.assert_close(sample["x_rgb"][:, 0, 0], torch.tensor([200, 100, 40]) / 255)
    torch.testing.assert_close(sample["x_ict"][0, 1:4], torch.tensor([2., 0., 1.]))
    torch.testing.assert_close(sample["y_action"][:, :3], torch.tensor([[3., 0., 1.], [4., 0., 1.]]))


def test_split_reserves_whole_first_session(tmp_path):
    from training.FlowMatchingTrainer import resolve_data_sources

    first = _session(tmp_path / "data", "session_000")
    second = _session(tmp_path / "data", "session_001")
    train, heldout = resolve_data_sources({"demos": 1}, str(tmp_path / "data"), "serve_bread", "demos")
    assert train == [str(second)]
    assert heldout == [str(first)]


def test_step_budget_updates_real_model_and_reports_heldout_loss(tmp_path):
    from training.FlowMatchingDataloader import FlowMatchingDataloader, SessionSpec
    from training.FlowMatchingModel import FlowMatchingModel
    from training.FlowMatchingTrainer import TrainConfig, train_one_epoch, eval_flow_loss
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)
    train = _session(tmp_path / "data", "session_001")
    heldout = _session(tmp_path / "data", "session_000")
    loaders = [DataLoader(FlowMatchingDataloader(
        [SessionSpec(str(session))], image_size=(16, 16), pred_horizon=2,
        single_hand=True, enable_augmentation=False,
        use_aux_obj_dynamics=False, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=False, use_pcd_features=False,
    ), batch_size=1) for session in (train, heldout)]
    model = FlowMatchingModel(
        single_hand=True, pred_horizon=2, max_ict=8, img_size=(16, 16),
        patch_size=8, vision_embed_dim=32, num_decoder_layers=1, num_heads=4,
        use_pcd_features=False, use_aux_obj_dynamics=False,
        use_aux_visual_foresight=False, use_aux_temporal_contrastive=False,
    )
    cfg = TrainConfig(device="cpu", use_amp=False, max_steps=1, use_ema=False,
                      single_hand=True, pred_horizon=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    metrics, step = train_one_epoch(model, loaders[0], optimizer, None, cfg, 1, 0, 1)
    assert step == 1
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    heldout_metrics = eval_flow_loss(model, loaders[1], cfg, max_batches=1)
    assert heldout_metrics["batches"] == 1
    assert np.isfinite(heldout_metrics["heldout_loss"])


def test_training_refuses_to_write_packaged_pretrained_output(tmp_path, monkeypatch):
    import training.FlowMatchingTrainer as trainer

    monkeypatch.setattr(trainer, "BUNDLE_ROOT", tmp_path)
    protected = tmp_path / "runs" / "serve_bread" / "pretrained"
    protected.mkdir(parents=True)
    checkpoint = protected / "latest.pt"
    checkpoint.write_bytes(b"released-model")
    cfg = trainer.TrainConfig(out_dir=str(protected), SESSIONS_TRAIN=[], SESSIONS_EVAL=[])
    with pytest.raises(ValueError, match="pretrained"):
        trainer.main(cfg)
    assert checkpoint.read_bytes() == b"released-model"


def test_smoke_artifacts_use_relocatable_bundle_paths(tmp_path, monkeypatch):
    import training.FlowMatchingTrainer as trainer

    monkeypatch.setattr(trainer, "BUNDLE_ROOT", tmp_path)
    _session(tmp_path / "data", "session_000")
    _session(tmp_path / "data", "session_001")
    cfg = trainer.TrainConfig(
        data_sources={"demos": 1}, max_steps=1, eval_batches=1,
        device="cpu", image_size=(16, 16), pred_horizon=2,
        batch_size=1, num_workers=0, single_hand=True,
        patch_size=8, vision_embed_dim=32, num_decoder_layers=1, num_heads=4,
        enable_augmentation=True, enable_aug_img=False, enable_aug_rrc=False,
        enable_aug_cutout=False, enable_aug_target_jittering=False,
        enable_aug_interpolation=True, use_legacy_rng=True,
        use_amp=False, use_ema=False,
    )
    result = trainer.main(cfg)
    directory = tmp_path / "runs" / "serve_bread" / "Human2Robot"
    saved = json.loads((directory / "smoke_config.json").read_text())
    assert saved["out_dir"] == "runs/serve_bread/Human2Robot"
    assert saved["data_root"] == "data"
    assert saved["SESSIONS_TRAIN"] == ["data/serve_bread/demos/session_001"]
    assert saved["SESSIONS_EVAL"] == ["data/serve_bread/demos/session_000"]
    checkpoint = torch.load(directory / "smoke.pt", map_location="cpu", weights_only=True)
    assert checkpoint["cfg"] == saved
    assert result["train_sessions"] == saved["SESSIONS_TRAIN"]
    assert result["eval_sessions"] == saved["SESSIONS_EVAL"]
    cached_result = trainer.main(cfg)
    assert cached_result["train"] == result["train"]
    assert cached_result["heldout"] == result["heldout"]


def _target_dataset(session, **kwargs):
    from training.FlowMatchingDataloader import FlowMatchingDataloader, SessionSpec

    return FlowMatchingDataloader(
        [SessionSpec(str(session))], image_size=(16, 16), pred_horizon=2,
        single_hand=True, frame_mode="camera_frame", enable_augmentation=False,
        use_aux_obj_dynamics=True, use_aux_visual_foresight=False,
        use_aux_temporal_contrastive=True, disable_kinematic_latching=True,
        **kwargs,
    )


@pytest.mark.parametrize("sample_index", [1, 2])
@pytest.mark.parametrize("action_mode", ["absolute", "delta"])
def test_finished_tail_repeats_measured_hand_object_rotation_and_done(tmp_path, sample_index, action_mode):
    session = _session(tmp_path / "data", "session_001")
    dataset = _target_dataset(session, action_mode=action_mode)
    assert len(dataset) == 3
    sample = dataset[sample_index]
    position = [4., 0., 1.] if action_mode == "absolute" else [0., 0., 0.]
    object_position = [1., 0., 1.] if action_mode == "absolute" else [0., 0., 0.]
    identity_rotation = [1., 0., 0., 1., 0., 0.]
    expected_hand = torch.tensor([position + identity_rotation + [1.]] * 2)
    expected_object = torch.tensor([object_position + identity_rotation] * 2)
    torch.testing.assert_close(sample["y_action"], expected_hand)
    torch.testing.assert_close(sample["y_obj_action"], expected_object)
    torch.testing.assert_close(sample["y_done"], torch.ones(2, 1))
    torch.testing.assert_close(sample["x_ict_future"][0, 1:4], torch.tensor([4., 0., 1.]))


@pytest.mark.parametrize("gap", ["missing_json", "missing_hand", "missing_done", "missing_object"])
def test_windows_crossing_tracking_or_label_gaps_are_excluded(tmp_path, gap):
    session = _session(tmp_path / "data", "session_001")
    middle = session / "preprocess" / "all_data" / "00102" / "training_data.json"
    if gap == "missing_json":
        middle.unlink()
    else:
        data = json.loads(middle.read_text())
        if gap == "missing_hand":
            del data["entities"]["hands"]["right"]
        elif gap == "missing_done":
            del data["metadata"]["is_finished"]
        else:
            del data["entities"]["objects"]["obj1"]
        middle.write_text(json.dumps(data))
    dataset = _target_dataset(session)
    assert [Path(path).parent.name for path in dataset.samples] == ["00103"]
    torch.testing.assert_close(dataset[0]["y_done"], torch.ones(2, 1))


def test_unfinished_end_has_no_invented_future_or_done_labels(tmp_path):
    session = _session(tmp_path / "data", "session_001")
    last = session / "preprocess" / "all_data" / "00103" / "training_data.json"
    data = json.loads(last.read_text())
    data["metadata"]["is_finished"] = 0.0
    last.write_text(json.dumps(data))
    dataset = _target_dataset(session)
    assert [Path(path).parent.name for path in dataset.samples] == ["00101"]
    torch.testing.assert_close(dataset[0]["y_done"], torch.zeros(2, 1))
    torch.testing.assert_close(dataset[0]["y_action"][:, :3], torch.tensor([[3., 0., 1.], [4., 0., 1.]]))
