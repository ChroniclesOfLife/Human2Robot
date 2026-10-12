"""CPU checks for the portable learned-policy inference boundary."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_recorded_observation_uses_reference_camera_transform(tmp_path):
    """Using the moving capture camera instead of cam0 changes learned ICTs."""
    import cv2
    from inference.observation import load_recorded_observation

    camera = np.eye(4)
    camera[0, 3] = 1.0
    moving_camera = np.eye(4)
    moving_camera[0, 3] = 8.0
    hand = np.eye(4)
    hand[0, 3] = 1.3
    obj = np.eye(4)
    obj[0, 3] = 1.5
    record = {
        "metadata": {"k": np.eye(3).tolist(), "c2w": moving_camera.tolist(),
                     "world_transforms": {"cam0": camera.tolist()}, "anchor_key": "obj1"},
        "entities": {"hands": {"right": {"T_hand_to_world": hand.tolist(), "grasp": 0.8}},
                     "objects": {"obj1": {"T_obj_to_world": obj.tolist()}}},
        "obs": {},
    }
    (tmp_path / "training_data.json").write_text(json.dumps(record))
    cv2.imwrite(str(tmp_path / "observation.png"), np.zeros((8, 12, 3), np.uint8))
    observation = load_recorded_observation(tmp_path)
    assert observation.hands_in_cam["right"][0, 3] == pytest.approx(0.3)
    assert observation.objects["obj1"].T_in_cam[0, 3] == pytest.approx(0.5)
    assert observation.grippers["right"] == 1.0
    assert observation.rgb_bgr.shape == (8, 12, 3)


def test_physical_success_requires_lift_release_and_settling():
    """A policy done flag or proximity while grasping cannot claim placement."""
    from inference.result import PhysicalSuccessTracker

    bread, plate = np.eye(4), np.eye(4)
    bread[2, 3] = plate[2, 3] = 0.9
    tracker = PhysicalSuccessTracker(bread, minimum_release_seconds=0.3, settled_seconds=0.2)
    assert not tracker.observe(bread, plate, gripper_closed=0.0, attached=False, dt=0.1)["physical_success"]
    bread[2, 3] += 0.04
    assert not tracker.observe(bread, plate, gripper_closed=1.0, attached=True, dt=0.1)["physical_success"]
    bread[2, 3] -= 0.04
    assert not tracker.observe(bread, plate, gripper_closed=0.0, attached=False, dt=0.1)["physical_success"]
    for _ in range(4):
        metrics = tracker.observe(bread, plate, gripper_closed=0.0, attached=False, dt=0.1)
    assert metrics["physical_success"] is True
    assert metrics["bread_lift_m"] == pytest.approx(0.04)
    bread[0, 3] += 0.05
    assert not tracker.observe(bread, plate, gripper_closed=0.0, attached=False, dt=0.1)["physical_success"]


def test_nonfinite_result_is_json_and_never_success():
    from inference.result import sanitize_result

    result = sanitize_result({"physical_success": True, "termination_reason": "physical_success",
                              "pose": [float("nan"), np.float32(float("inf"))]})
    assert result["physical_success"] is False
    assert result["pose"] == [None, None]
    json.dumps(result, allow_nan=False)


def test_serve_bread_missing_checkpoint_returns_failure_before_isaac(tmp_path):
    command = [sys.executable, str(ROOT / "inference" / "serve_bread.py"),
               "--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(tmp_path)]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode != 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["physical_success"] is False
    assert result["termination_reason"] == "exception:FileNotFoundError"
    assert result["perception_mode"] == "simulator_ground_truth"


def test_packaged_checkpoint_runs_actual_neural_inference_on_cpu():
    """The released checkpoint must load strictly and produce finite trajectories."""
    from inference.observation import first_recorded_frame, load_recorded_observation
    from inference.policy import ICTPolicy
    import torch
    torch.set_num_threads(4)

    checkpoint = ROOT / "runs" / "serve_bread" / "pretrained" / "latest.pt"
    if not checkpoint.exists():
        pytest.skip("packaged checkpoint is not available yet")
    observation = load_recorded_observation(first_recorded_frame(ROOT / "data" / "serve_bread" / "demos" / "session_000" / "preprocess" / "all_data"))
    policy = ICTPolicy({"ckpt": str(checkpoint), "num_inference_steps": 2, "seed": 7}, device="cpu")
    tokens, mask = policy.build_ict(observation.hands_in_cam, observation.grippers, observation.objects, observation.anchor_key)
    anchor = observation.objects[observation.anchor_key]
    trajectory, done = policy.infer(policy.prepare_image(observation.rgb_bgr), tokens, mask,
                                    policy.compute_anchor_uv(observation.image_anchor, observation.K, observation.rgb_bgr.shape[1], observation.rgb_bgr.shape[0]))
    positions, rotations, grasps = trajectory["right"]
    assert positions.shape == (50, 3)
    assert rotations.shape == (50, 6)
    assert grasps.shape == (50, 1)
    assert np.isfinite(positions).all() and np.isfinite(rotations).all()
    assert np.all((grasps >= 0) & (grasps <= 1))
    assert 0 <= done <= 1
    assert not np.array_equal(positions, policy.noise[0, :, :3].cpu().numpy())


def test_policy_rejects_missing_model_weights(tmp_path):
    """A partially compatible checkpoint must fail rather than run random layers."""
    import torch
    from training.FlowMatchingModel import FlowMatchingModel
    from inference.policy import ICTPolicy

    cfg = {"single_hand": True, "pred_horizon": 3, "max_ict": 4,
           "image_size": [16, 16], "patch_size": 8, "vision_embed_dim": 24,
           "num_decoder_layers": 1, "num_heads": 4, "use_pcd_features": False,
           "use_aux_obj_dynamics": False, "use_aux_visual_foresight": False,
           "use_aux_temporal_contrastive": False, "use_region_attn": False}
    model = FlowMatchingModel(single_hand=True, pred_horizon=3, max_ict=4, img_size=(16, 16),
                              patch_size=8, vision_embed_dim=24, num_decoder_layers=1, num_heads=4,
                              use_pcd_features=False, use_aux_obj_dynamics=False,
                              use_aux_visual_foresight=False, use_aux_temporal_contrastive=False,
                              use_region_attn=False)
    state = model.state_dict()
    del state["state_proj.weight"]
    torch.save({"model": state}, tmp_path / "incomplete.pt")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "dataset_stats.json").write_text(json.dumps({"pos": {"mean": [0, 0, 0], "std": [1, 1, 1]}}))
    with pytest.raises(RuntimeError, match="Missing key"):
        ICTPolicy({"ckpt": str(tmp_path / "incomplete.pt")}, device="cpu")


def test_wxai_gripper_commands_and_feedback_share_measured_closed_width():
    """Closing must command 22 mm and report closed, rather than target zero."""
    from inference.wxai_trossen_arm import IsaacWXAITrossenArm

    class Array:
        def __init__(self, value):
            self.value = np.asarray(value)

        def numpy(self):
            return self.value

    class NativeController:
        gripper_dof_index = 6
        width = 0.044

        def get_world_poses(self):
            return Array([[0, 0, 0]]), Array([[1, 0, 0, 0]])

        def get_dof_positions(self):
            return Array([[0, 0, 0, 0, 0, 0, self.width, self.width]])

        def set_gripper_position(self, width):
            self.width = width

    class Simulation:
        def update(self):
            pass

    native = NativeController()
    arm = IsaacWXAITrossenArm(native, Simulation(), np.eye(4))
    arm.set_gripper(1.0)
    assert native.width == pytest.approx(0.022)
    assert arm.get_gripper() == pytest.approx(1.0)
    arm.set_gripper(0.5)
    assert native.width == pytest.approx(0.033)
    assert arm.get_gripper() == pytest.approx(0.5)


def test_simulation_config_selects_requested_gpu_without_conflicting_sdk_fields():
    """SimulationApp rejects simultaneous active_gpu and active_cuda_gpus."""
    from inference.serve_bread import parse_args, simulation_config

    settings = simulation_config(parse_args(["--headless", "--gpu", "3", "--width", "320", "--height", "180"]))
    assert settings["active_cuda_gpus"] == [3]
    assert settings["physics_gpu"] == 3
    assert "active_gpu" not in settings
    assert settings["multi_gpu"] is False
    assert settings["width"] == 320 and settings["height"] == 180


def test_flow_grasp_values_decode_as_probabilities_not_logits():
    """Training flow targets encode grasp as 0/1; sigmoid would close an open hand."""
    import torch
    from inference.policy import ICTPolicy

    class ControlledVelocity(torch.nn.Module):
        def forward(self, **kwargs):
            velocity = torch.zeros_like(kwargs["x_t"])
            velocity[0, :, 9] = torch.tensor([0.1, 0.9])
            return {"v_pred": velocity, "done_logit": torch.zeros((1, 1))}

    policy = object.__new__(ICTPolicy)
    policy.model = ControlledVelocity()
    policy.noise = torch.zeros((1, 2, 10))
    policy.device = "cpu"
    policy.num_inference_steps = 1
    policy.single_hand = True
    policy.single_hand_side = "right"
    policy.use_done_in_flow = False
    trajectory, _ = policy.infer(torch.zeros((1, 3, 8, 8)), torch.zeros((1, 8, 20)), torch.ones((1, 8), dtype=torch.bool))
    grasp = trajectory["right"][2].reshape(-1)
    np.testing.assert_allclose(grasp, [0.1, 0.9], atol=1e-7)
    assert (grasp > 0.5).tolist() == [False, True]


def test_result_is_written_before_sdk_shutdown_exits_process(tmp_path, monkeypatch):
    """Kit fast shutdown must not erase exception reports or replace failure with exit 0."""
    import inference.serve_bread as entry

    for name in ("checkpoint.pt", "config.json", "dataset_stats.json", "scene.usda", "calibration.npz"):
        (tmp_path / name).write_text("{}")

    class FastShutdown:
        def close(self, *, exit_code):
            raise SystemExit(exit_code)

    def failed_runtime(args, result, resources):
        resources["simulation_app"] = FastShutdown()
        raise RuntimeError("neural checkpoint failed after physics startup")

    monkeypatch.setattr(entry, "_run", failed_runtime)
    output = tmp_path / "result"
    with pytest.raises(SystemExit) as shutdown:
        entry.main(["--checkpoint", str(tmp_path / "checkpoint.pt"), "--scene", str(tmp_path / "scene.usda"),
                    "--calibration", str(tmp_path / "calibration.npz"), "--output-dir", str(output)])
    assert shutdown.value.code == 1
    result = json.loads((output / "result.json").read_text())
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert result["physical_success"] is False
    assert result["termination_reason"] == "exception:RuntimeError"
    assert "neural checkpoint failed" in (output / "exception.txt").read_text()
    assert manifest["status"] == "failed"
