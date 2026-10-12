"""Human2Robot learned Flow Matching / ICT serve-bread loop in Isaac Sim.

Run with Isaac Sim's Python: python inference/serve_bread.py --headless --gpu 1.
Object perception is simulator ground truth with calibrated local PCA frames.
The clean RGB observation hides the simulated robot and overlays training-style
virtual hands and object keypoints. This entry point does not run a real-camera
detector or inpainting pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--policy-device", help="Defaults to cuda:<gpu>; use cpu for CPU inference")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/serve_bread/pretrained/latest.pt")
    parser.add_argument("--scene", type=Path, default=ROOT / "assets/serve_bread.usda")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration/object_keypoints.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "cfg/inference/serve_bread.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/serve_bread")
    parser.add_argument("--max-replans", type=int, default=45)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--exec-horizon", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strict-contact", action="store_true", help="Disable the distance-gated physical FixedJoint grasp assist")
    parser.add_argument("--attach-distance", type=float, default=0.08)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


def simulation_config(args):
    """SDK launch config: renderer GPU list and PhysX GPU use the same device."""
    return {"headless": args.headless, "width": args.width, "height": args.height,
            "multi_gpu": False, "active_cuda_gpus": [args.gpu],
            "physics_gpu": args.gpu, "renderer": "RaytracedLighting"}


def main(argv=None):
    from inference.paths import display_path, portable_record, portable_text
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = {"project": "Human2Robot", "run_id": run_id,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "inference_mode": "learned_flow_matching", "physical_success": False,
              "termination_reason": "initialization", "replans": 0,
              "perception_mode": "simulator_ground_truth",
              "clean_background_mode": "simulator_robot_visibility",
              "grasp_mode": "strict_contact_only" if args.strict_contact else "distance_gated_assisted_fixed_joint",
              "grasp_assist_enabled": not args.strict_contact,
              "checkpoint": display_path(args.checkpoint), "scene": display_path(args.scene),
              "renderer_gpu": args.gpu, "physics_gpu": args.gpu,
              "policy_device": args.policy_device or f"cuda:{args.gpu}",
              "capture_resolution": [args.width, args.height], "seed": args.seed,
              "summary_errors": []}
    manifest = output / "run_manifest.json"
    manifest.write_text(json.dumps({**result, "status": "running"}, indent=2) + "\n")
    # Keep a failed/incomplete record even if a native SDK process aborts before
    # Python can run the final writer; a previous successful run cannot leak in.
    (output / "result.json").write_text(json.dumps({**result, "termination_reason": "incomplete_run"}, indent=2) + "\n")
    resources = {}
    try:
        if min(args.width, args.height, args.max_replans, args.exec_horizon, args.cpu_threads) <= 0 or args.gpu < 0 or args.attach_distance <= 0:
            raise ValueError("Resolution, horizons, CPU threads and attachment distance must be positive; GPU must be nonnegative")
        for path in (args.checkpoint, args.checkpoint.parent / "config.json",
                     args.checkpoint.parent / "dataset_stats.json", args.scene, args.calibration):
            if not path.is_file():
                raise FileNotFoundError(f"Missing required bundle input: {path}")
        _run(args, result, resources)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result.update(physical_success=False, termination_reason=f"exception:{type(error).__name__}",
                      error=str(error))
        (output / "exception.txt").write_text(portable_text(traceback.format_exc()), encoding="utf-8")
        traceback.print_exc()
    finally:
        from inference.result import sanitize_result
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result = portable_record(sanitize_result(result))
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest.write_text(json.dumps({**result, "status": "completed" if result["physical_success"] else "failed"}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Physical success: {result['physical_success']}; termination: {result['termination_reason']}")
    exit_code = 0 if result["physical_success"] else 1
    # Kit's fast shutdown can exit the entire process. Persist reports first,
    # and preserve a failed episode's nonzero process status at shutdown.
    if resources.get("simulation_app") is not None:
        resources["simulation_app"].close(exit_code=exit_code)
    return exit_code


def _run(args, result, resources):
    # Resolve this repository's lightweight package before the SDK adds its
    # own search paths. No neural/CUDA modules are imported until physics starts.
    import utils
    from isaacsim import SimulationApp
    simulation_app = None
    arm = camera = trace = None
    try:
        simulation_app = SimulationApp(simulation_config(args))
        resources["simulation_app"] = simulation_app
        import carb
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import PhysxSchema, UsdGeom, UsdPhysics

        settings = carb.settings.get_settings()
        settings.set_int("/physics/cudaDevice", args.gpu)
        ctx = omni.usd.get_context()
        ctx.open_stage(str(args.scene.resolve()))
        for _ in range(30):
            simulation_app.update()
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not open the requested scene")
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(100000)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physics_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                physics_api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(16384)
        cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
        robot_path = cfg.get("robot_path", "/World/wxai_robot")
        camera_path = cfg.get("camera_path", "/World/rsd455/RSD455/Camera_OmniVision_OV9782_Color")
        bread_path = cfg.get("bread_path", "/World/objaverse_food_produce_baguette_v01_00")
        plate_path = cfg.get("plate_path", "/World/sm_dishware_plastic_bigPlate_a01_01")
        for path in (robot_path, camera_path, bread_path, plate_path):
            if not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Required scene prim is absent: {path}")
        # Scene calibration is measured geometry, not an action sequence.
        def usd_world_pose(path):
            matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
            pose = np.asarray(matrix, dtype=np.float64).T
            # Cache local offsets are metric rigid frames; mesh scale is not part of SE(3).
            u, _, vt = np.linalg.svd(pose[:3, :3])
            correction = np.diag([1., 1., np.linalg.det(u @ vt)])
            pose[:3, :3] = u @ correction @ vt
            return pose

        camera_cfg = cfg.get("camera", {})
        T_usd_camera_world = usd_world_pose(camera_path)
        optical_to_usd = np.diag([1.0, -1.0, -1.0, 1.0])
        T_world_to_cam = np.asarray(camera_cfg.get("T_world_to_cam", np.linalg.inv(T_usd_camera_world @ optical_to_usd)), np.float64)
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
        focal = float(usd_camera.GetFocalLengthAttr().Get())
        horizontal = float(usd_camera.GetHorizontalApertureAttr().Get())
        vertical = float(usd_camera.GetVerticalApertureAttr().Get())
        K = np.array([[focal / horizontal * args.width, 0, args.width / 2],
                      [0, focal / vertical * args.height, args.height / 2], [0, 0, 1]], dtype=np.float64)
        if "K" in camera_cfg:
            K = np.asarray(camera_cfg["K"], dtype=np.float64)
            original_width, original_height = camera_cfg["resolution"]
            K[0, :] *= args.width / original_width
            K[1, :] *= args.height / original_height
        if T_world_to_cam.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(T_world_to_cam).all() or not np.isfinite(K).all():
            raise ValueError("Camera calibration must contain finite 4x4 extrinsics and 3x3 intrinsics")
        result.update(camera_K=K.tolist(), T_world_to_cam=T_world_to_cam.tolist(),
                      camera_calibration_source="config_with_usd_defaults" if camera_cfg else "usd_camera_pose_and_aperture",
                      camera_intrinsics_source="config_scaled_to_capture" if "K" in camera_cfg else "usd_camera_aperture",
                      camera_extrinsics_source="config" if "T_world_to_cam" in camera_cfg else "usd_camera_pose")
        T_cam_to_world = np.linalg.inv(T_world_to_cam)

        from inference.wxai_grasp import AssistedBreadGrasp, apply_grasp_friction, ensure_wxai_tcp_site, prepare_bread_physics
        bread_body_path = prepare_bread_physics(stage, bread_path)
        result["grasp_site"] = ensure_wxai_tcp_site(stage, robot_path)
        result["friction_bindings"] = apply_grasp_friction(stage, bread_path, robot_path)
        SimulationManager.switch_physics_engine("physx")
        SimulationManager.set_device(f"cuda:{args.gpu}")
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(60):
            simulation_app.update()
        # Start physics before importing the neural CUDA stack.
        import torch
        from inference.controller import TrajectoryController
        from inference.interfaces import Frame, ObjectState
        from inference.policy import ICTPolicy
        from inference.result import PhysicalSuccessTracker
        from inference.wxai_controller import RobotType, TrossenAIController
        from inference.wxai_trace import ControlTraceWriter
        from inference.wxai_trossen_arm import IsaacWXAITrossenArm

        torch.set_num_threads(args.cpu_threads)
        trace = ControlTraceWriter(args.output_dir / "controller_trace.csv")
        trace_context = {"replan": 0, "step": 0}
        def record_trace(sample):
            trace.append(sample, phase=f"learned_replan_{trace_context['replan']:03d}", phase_step=trace_context["step"])
            trace_context["step"] += 1

        native = TrossenAIController(robot_path=robot_path, robot_type=RobotType.WXAI,
                                    arm_dof_indices=cfg.get("arm_dof_indices", [0, 1, 2, 3, 4, 5]),
                                    gripper_dof_index=cfg.get("gripper_dof_index", 6),
                                    default_dof_positions=cfg.get("default_joint_positions", [0., 0., 0., 0., 0., 0., .044, .044]),
                                    ik_scale=0.35, ik_damping=0.08, max_joint_delta=0.02,
                                    telemetry_sink=record_trace)
        policy_ee_offset = np.asarray(cfg.get("policy_ee_offset", [.1055, 0., 0.]), np.float64)
        native.ee_offset = policy_ee_offset
        result["policy_ee_offset_m"] = policy_ee_offset.tolist()
        result["controller_safety"] = {"ik_scale": .35, "ik_damping": .08, "max_joint_delta_rad": .02}
        arm = IsaacWXAITrossenArm(native, simulation_app, T_world_to_cam, dt=1 / 60)
        assist = AssistedBreadGrasp(stage, bread_body_path, f"{robot_path}/link_6", arm.get_T_ee_in_world,
                                   args.output_dir, attach_distance=args.attach_distance, enabled=not args.strict_contact)
        arm.grasp_assist = assist
        print(f"Grasp mode: {result['grasp_mode']}; renderer/physics GPU: {args.gpu}")

        with np.load(args.calibration, allow_pickle=False) as cache:
            local_frames = {"obj1": np.linalg.inv(cache["T_bread_current"]) @ cache["T_bread_pca1"],
                            "obj2": np.linalg.inv(cache["T_plate_current"]) @ cache["T_plate_pca1"]}
            local_points = {}
            for key, label in (("obj1", "bread"), ("obj2", "plate")):
                inverse = np.linalg.inv(cache[f"T_{label}_pca1"])
                points = cache[f"{label}_pts_cam"]
                local_points[key] = (inverse[:3, :3] @ points.T).T + inverse[:3, 3]
        # Current/PCA cache transforms share an optical frame; their ratio is object-local.
        # Correct a nested rigid-body frame using the actual initial root/body relationship.
        bread_body_to_root = np.linalg.inv(assist.bread_world_pose()) @ usd_world_pose(bread_path)
        plate_body = RigidPrim(plate_path) if stage.GetPrimAtPath(plate_path).HasAPI(UsdPhysics.RigidBodyAPI) else None
        def measured_plate_root():
            if plate_body is None:
                return usd_world_pose(plate_path)
            position, quaternion = assist._world_pose(plate_body)
            from inference.wxai_grasp import _pose_matrix
            return _pose_matrix(position, quaternion)
        def object_world_poses():
            return {"obj1": assist.bread_world_pose() @ bread_body_to_root @ local_frames["obj1"],
                    "obj2": measured_plate_root() @ local_frames["obj2"]}
        def observed_objects():
            return {key: ObjectState((T_world_to_cam @ pose).astype(np.float32), local_points[key].astype(np.float32))
                    for key, pose in object_world_poses().items()}

        initial = object_world_poses()
        tracker = PhysicalSuccessTracker(initial["obj1"])
        def sample_physics():
            poses = object_world_poses()
            plate_points_world = local_points["obj2"] @ poses["obj2"][:3, :3].T + poses["obj2"][:3, 3]
            result.update(tracker.observe(poses["obj1"], poses["obj2"], gripper_closed=arm.get_gripper(),
                                         attached=assist.attached, dt=arm.dt, plate_points_world=plate_points_world))
        arm.step_callback = sample_physics

        class IsaacCamera:
            def __init__(self):
                self.product = rep.create.render_product(camera_path, (args.width, args.height))
                self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
                self.rgb.attach([self.product])
                self.depth.attach([self.product])
                for _ in range(10):
                    arm.advance()

            def grab_bgr(self, step):
                for _ in range(2):
                    step()
                data = np.asarray(self.rgb.get_data())
                if data.ndim != 3 or data.shape[-1] < 3:
                    raise RuntimeError("Isaac camera returned an empty RGB frame")
                image = data[..., :3]
                if image.dtype != np.uint8:
                    image = np.clip(image * (255 if image.max() <= 1 else 1), 0, 255).astype(np.uint8)
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            def get_frame(self):
                image = self.grab_bgr(arm.advance)
                depth = np.asarray(self.depth.get_data(), dtype=np.float32).squeeze()
                if depth.shape != (args.height, args.width):
                    raise RuntimeError("Isaac camera returned an invalid optical depth frame")
                return Frame(image, depth, K.copy())

            def clean_background(self):
                imageable = UsdGeom.Imageable(stage.GetPrimAtPath(robot_path))
                visibility = imageable.GetVisibilityAttr().Get()
                timeline.pause()
                try:
                    imageable.MakeInvisible()
                    for _ in range(3):
                        simulation_app.update()
                    return self.grab_bgr(simulation_app.update)
                finally:
                    imageable.GetVisibilityAttr().Set(visibility)
                    for _ in range(2):
                        simulation_app.update()
                    timeline.play()

            def close(self):
                self.rgb.detach()
                self.depth.detach()
                self.product.destroy()

        camera = IsaacCamera()
        from inference.simulator_perception import draw_training_observation
        T_align = np.asarray(cfg.get("T_align", [[-1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]), np.float32)
        result["T_align"] = T_align.tolist()
        policy_cfg = {"ckpt": str(args.checkpoint.resolve()), "seed": args.seed}
        if args.num_inference_steps is not None:
            policy_cfg["num_inference_steps"] = args.num_inference_steps
        policy = ICTPolicy(policy_cfg, device=result["policy_device"])
        if policy.sides != ["right"]:
            raise ValueError("The packaged WXAI scene adapter requires a right single-hand checkpoint")
        result.update(checkpoint_loaded_strict=True, num_inference_steps=policy.num_inference_steps,
                      checkpoint_metadata=policy.checkpoint_metadata,
                      neural_forward_evaluations=0)
        controller = TrajectoryController({"right": arm}, cfg.get("control", {}))
        result["termination_reason"] = "max_replans_without_physical_success"
        for replan in range(1, args.max_replans + 1):
            if not simulation_app.is_running():
                result["termination_reason"] = "simulation_closed"
                break
            result["replans"] = replan
            trace_context.update(replan=replan, step=0)
            frame = camera.get_frame()
            hands = {"right": arm.get_T_ee_in_cam() @ T_align}
            grippers = {"right": arm.get_gripper()}
            objects = observed_objects()
            anchor = objects["obj1"]
            clean = draw_training_observation(camera.clean_background(), K, hands, grippers, objects)
            tokens, mask = policy.build_ict(hands, grippers, objects, "obj1")
            x_rgb = policy.prepare_image(clean)
            uv = policy.compute_anchor_uv(anchor, K, args.width, args.height)
            prediction, done = policy.infer(x_rgb, tokens, mask, uv)
            result["neural_forward_evaluations"] += policy.num_inference_steps
            result["last_done_probability"] = float(done)
            pos, o6d, grasp = prediction["right"]
            targets = np.asarray([policy.decode_ee_in_cam(p, rot, anchor, T_align) for p, rot in zip(pos, o6d)])
            plate_pose_world = T_cam_to_world @ objects["obj2"].T_in_cam
            plate_points_world = local_points["obj2"] @ plate_pose_world[:3, :3].T + plate_pose_world[:3, 3]
            np.savez_compressed(args.output_dir / f"replan_{replan:03d}.npz", input_rgb_bgr=frame.rgb,
                                input_depth_m=frame.depth_m, clean_rgb_bgr=clean,
                                ict=tokens.cpu().numpy(), ict_mask=mask.cpu().numpy(),
                                network_rgb=x_rgb.cpu().numpy(), anchor_uv=uv.cpu().numpy() if uv is not None else np.empty((0, 2)),
                                initial_noise=policy.noise.cpu().numpy(), K=K, T_world_to_cam=T_world_to_cam,
                                T_hand_in_cam=hands["right"], T_bread_in_cam=objects["obj1"].T_in_cam,
                                T_plate_in_cam=objects["obj2"].T_in_cam, gripper_closed=grippers["right"],
                                plate_points_world=plate_points_world,
                                predicted_position=pos, predicted_o6d=o6d, predicted_grasp=grasp,
                                decoded_ee_targets_in_cam=targets, done_probability=done)
            cv2.imwrite(str(args.output_dir / "latest_clean_input.png"), clean)
            controller.execute_chunk({"right": targets}, {"right": grasp}, dt=cfg.get("control_dt", .1),
                                     n_steps=min(args.exec_horizon, len(targets)))
            print(f"Replan {replan}: done={done:.3f}, lift={tracker.lift_m:.3f} m, placement={tracker.metrics.get('bread_to_plate_m'):.3f} m")
            if tracker.metrics["physical_success"]:
                result["termination_reason"] = "physical_success"
                break
        result.update(tracker.metrics, grasp_assist_used=assist.assist_used,
                      grasp_attach_count=assist.attach_count, grasp_release_count=assist.release_count,
                      physics_attached_at_end=assist.attached,
                      final_ee_cam_xyz=arm.get_T_ee_in_cam()[:3, 3].tolist(),
                      final_gripper_closed=arm.get_gripper())
    finally:
        # Record physical attachment before releasing runtime resources.
        for label, resource in (("camera", camera), ("arm", arm), ("trace", trace)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result["summary_errors"].append(f"{label}:{type(error).__name__}:{error}")
        if result["summary_errors"]:
            result.update(physical_success=False, termination_reason="cleanup_failed")


if __name__ == "__main__":
    raise SystemExit(main())
