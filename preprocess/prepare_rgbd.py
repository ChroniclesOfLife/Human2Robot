"""Export measured fixed RGB-D serve_bread demonstrations for policy training.

Recording layout: rgb/<stem>.png, depth/<stem>.npy (float metres),
intrinsics.json ({"k": [[...], [...], [...]], "fps": 30}), and
labels.json ({"<stem>": false/true}) with explicit task completion labels.
Optional JSON hand annotations live in hands/<stem>.json; annotated object
masks live in masks/<stem>/{obj1,obj2,arm}.png. PNG depth requires a declared
metres-per-unit scale. Optional timestamps in intrinsics.json map stems to
seconds. All depth must already be aligned to a rectified RGB image.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from preprocess.geometry import PoseSmoother, hand_from_keypoints, object_from_mask
from preprocess.models import (GroundedSamSegmenter, HaMeRHandTracker, Inpainter,
                               JsonHandTracker, MaskObjectSegmenter, MediaPipeHandTracker)
from preprocess.visualize import render_observation


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, data):
    path.write_text(json.dumps(_json_safe(data), indent=2, allow_nan=False) + "\n")


def load_depth(recording, stem, *, scale=None):
    path = recording / "depth" / f"{stem}.npy"
    if path.is_file():
        depth = np.load(path, allow_pickle=False)
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"NPY depth must be floating-point metres: {path}")
    else:
        path = recording / "depth" / f"{stem}.png"
        if not path.is_file():
            raise FileNotFoundError(f"No aligned depth for frame {stem}")
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError("PNG depth requires --depth-scale or intrinsics.json depth_scale (metres/unit)")
        depth = np.asarray(Image.open(path), dtype=float) * scale
    if depth.ndim != 2:
        raise ValueError(f"Depth must have shape HxW: {path}")
    return depth


def _backends(args, recording):
    if args.hand_backend == "json":
        tracker = JsonHandTracker(args.keypoints_dir or recording / "hands")
    else:
        tracker = MediaPipeHandTracker(args.hand_model, flip_handedness=args.flip_handedness)
        if args.hand_backend == "hamer":
            tracker = HaMeRHandTracker(tracker, args.hamer_checkpoint, device=args.device)
    if args.object_backend == "masks":
        segmenter = MaskObjectSegmenter(args.masks_dir or recording / "masks")
    else:
        segmenter = GroundedSamSegmenter(dino_model=args.dino_model, sam_checkpoint=args.sam2_checkpoint,
                                        sam_config=args.sam2_config, allow_download=args.allow_download,
                                        device=args.device, box_threshold=args.box_threshold)
    inpainter = Inpainter(args.inpaint, args.lama_model, allow_download=args.allow_download)
    return tracker, segmenter, inpainter


def prepare(args):
    recording, output = Path(args.input).resolve(), Path(args.output).resolve()
    all_data = output / "preprocess" / "all_data"
    if all_data.exists() or (output / "preprocess" / "manifest.json").exists():
        raise FileExistsError(f"Choose a new output session; preprocessing already exists: {output}")
    info = json.loads((recording / "intrinsics.json").read_text())
    k, fps = np.asarray(info.get("k", info.get("K")), dtype=float), float(info["fps"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("intrinsics.json needs a finite 3x3 k and positive focal lengths")
    if not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("Use rectified pinhole intrinsics with bottom row [0, 0, 1]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    frames = sorted((recording / "rgb").glob("*.png"),
                    key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem))
    if not frames:
        raise ValueError(f"No RGB PNG sequence in {recording / 'rgb'}")
    labels_path = Path(args.labels_file) if args.labels_file else recording / "labels.json"
    labels = json.loads(labels_path.read_text())
    if not isinstance(labels, dict) or any(path.stem not in labels for path in frames):
        raise ValueError("labels.json must contain an explicit finish label for every RGB stem")
    if any(not isinstance(labels[path.stem], (bool, int, float)) or labels[path.stem] not in (0, 1) for path in frames):
        raise ValueError("Finish labels must be boolean or 0/1, not inferred from sequence length")
    scale = args.depth_scale if args.depth_scale is not None else info.get("depth_scale")
    if scale is not None:
        scale = float(scale)
    timestamps = info.get("timestamps")
    if timestamps is not None:
        if not isinstance(timestamps, dict) or any(path.stem not in timestamps for path in frames):
            raise ValueError("timestamps must map every RGB stem to seconds")
        times = [float(timestamps[path.stem]) for path in frames]
        if not np.isfinite(times).all() or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("RGB timestamps must be finite and strictly increasing")
    else:
        times = [i / fps for i in range(len(frames))]
    numeric_settings = [args.smoothing_alpha, args.min_depth, args.max_depth, args.motion_threshold,
                        args.box_threshold, args.hand_confidence]
    if not np.isfinite(numeric_settings).all():
        raise ValueError("Preprocessing numeric settings must be finite")
    if not 0 < args.smoothing_alpha <= 1 or args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("Invalid smoothing or depth bounds")
    if args.mask_dilation < 0 or args.motion_threshold < 0 or not 0 <= args.hand_confidence <= 1 or not 0 < args.box_threshold < 1:
        raise ValueError("Invalid dilation, motion, or confidence threshold")
    # Validate inexpensive inputs before optional model initialization.
    tracker, segmenter, inpainter = _backends(args, recording)
    smoother = PoseSmoother(args.smoothing_alpha)
    anchor, previous_grasp, previous_objects = None, {}, {}
    manifest = {"task": "serve_bread", "source": str(recording), "input_frames": len(frames),
                "accepted_frames": 0, "rejected_frames": [], "world_frame": "fixed_camera",
                "units": "metres", "finish_labels": str(labels_path),
                "settings": {"hand_backend": args.hand_backend, "object_backend": args.object_backend,
                             "inpaint": args.inpaint, "smoothing_alpha": args.smoothing_alpha,
                             "hand_origin": args.hand_origin, "depth_scale": scale,
                             "timestamp_source": "recording" if timestamps is not None else "fps",
                             "min_depth": args.min_depth, "max_depth": args.max_depth,
                             "mask_dilation": args.mask_dilation, "motion_threshold": args.motion_threshold,
                             "hand_confidence": args.hand_confidence, "box_threshold": args.box_threshold,
                             "device": args.device, "hand_model": args.hand_model,
                             "hamer_checkpoint": args.hamer_checkpoint, "dino_model": args.dino_model,
                             "sam2_checkpoint": args.sam2_checkpoint, "sam2_config": args.sam2_config,
                             "lama_model": getattr(inpainter, "model_path", args.lama_model),
                             "flip_handedness": args.flip_handedness}, "status": "processing"}
    identity = np.eye(4)
    all_data.mkdir(parents=True)
    try:
        for index, (rgb_path, timestamp) in enumerate(zip(frames, times)):
            stem = rgb_path.stem
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            h, w = rgb.shape[:2]
            depth = load_depth(recording, stem, scale=scale)
            if depth.shape != (h, w):
                raise ValueError(f"Depth and RGB are not aligned in frame {stem}")
            detections = tracker.detect(rgb, stem, (timestamp - times[0]) * 1000)
            hands = {}
            for side in ("right", "left"):
                detection = detections.get(side)
                hand = None
                if detection and np.isfinite(detection["confidence"]) and detection["confidence"] >= args.hand_confidence:
                    hand = hand_from_keypoints(detection["keypoints"], depth, k,
                                               previous_grasp=previous_grasp.get(side, 0.),
                                               origin=args.hand_origin, min_depth=args.min_depth,
                                               max_depth=args.max_depth)
                if hand is None:
                    smoother.reset(side)
                    previous_grasp.pop(side, None)
                else:
                    hand["T_hand_to_world"] = smoother.update(side, hand["T_hand_to_world"])
                    hand["confidence"] = detection["confidence"]
                    hands[side] = hand
                    previous_grasp[side] = hand["grasp"]
            masks = segmenter.segment(rgb, stem)
            objects, keypoints = {}, {}
            for name in ("obj1", "obj2"):
                previous = previous_objects.get(name)
                obj = object_from_mask(masks[name], depth, k,
                                       previous_rotation=previous[0][:3, :3] if previous else None,
                                       min_depth=args.min_depth, max_depth=args.max_depth)
                if obj is None:
                    smoother.reset(name)
                    previous_objects.pop(name, None)
                    continue
                measured = obj["T_obj_to_world"]
                moving = bool(previous and np.linalg.norm(measured[:3, 3] - previous[0][:3, 3]) /
                              (timestamp - previous[1]) > args.motion_threshold)
                previous_objects[name] = (measured.copy(), timestamp)
                obj["T_obj_to_world"] = smoother.update(name, measured)
                pose = obj["T_obj_to_world"]
                keypoints[name] = {"world": obj.pop("world"), "local": None}
                keypoints[name]["local"] = (keypoints[name]["world"] - pose[:3, 3]) @ pose[:3, :3]
                obj.pop("local")
                obj["is_dynamic"] = moving
                obj["label"] = "bread" if name == "obj1" else "plate"
                objects[name] = obj
            if "right" not in hands or set(objects) != {"obj1", "obj2"}:
                reason = "invalid_right_hand" if "right" not in hands else "missing_object_depth_or_mask"
                manifest["rejected_frames"].append({"idx": index, "source_frame": stem, "reason": reason})
                # Entire accepted sequence has a gap, so restart causal histories.
                for key in ("right", "left", "obj1", "obj2"):
                    smoother.reset(key)
                previous_grasp.clear()
                previous_objects.clear()
                continue
            protect_objects = masks["obj1"] | masks["obj2"]
            arm_mask = masks["arm"] & ~protect_objects
            if args.inpaint != "none" and not arm_mask.any():
                manifest["rejected_frames"].append({"idx": index, "source_frame": stem, "reason": "missing_arm_mask"})
                for key in ("right", "left", "obj1", "obj2"):
                    smoother.reset(key)
                previous_grasp.clear()
                previous_objects.clear()
                continue
            if args.mask_dilation:
                arm_mask = np.asarray(Image.fromarray(arm_mask.astype(np.uint8) * 255).filter(
                    ImageFilter.MaxFilter(2 * args.mask_dilation + 1))) > 0
                arm_mask &= ~protect_objects
            cleaned = inpainter.inpaint(rgb, arm_mask)
            observation = render_observation(cleaned, hands, keypoints, k)
            if anchor is None:
                anchor = objects["obj1"]["T_obj_to_world"].copy()
            frame_dir = all_data / f"{index:05d}"
            frame_dir.mkdir()
            for filename, array in (("rgb.png", rgb), ("rgb_clean.png", cleaned),
                                    ("observation.png", observation),
                                    ("mask_arm.png", arm_mask.astype(np.uint8) * 255)):
                Image.fromarray(array).save(frame_dir / filename)
            data = {"metadata": {"idx": index, "source_frame": stem, "ts": timestamp,
                                  "w": w, "h": h, "fps": fps, "k": k, "c2w": identity,
                                  "anchor_key": "obj1", "is_finished": float(labels[stem]),
                                  "world_transforms": {"cam0": identity, "virtual_static_anchor": anchor},
                                  "preprocessing": manifest["settings"]},
                    "obs": {"rgb_path": "rgb.png", "rgb_WoArm_path": "rgb_clean.png",
                            "mask_arm_path": "mask_arm.png", "observation_path": "observation.png",
                            "rgb_WoArm_WArmObjKpts_path": "observation.png", "objects_kpts": keypoints},
                    "entities": {"hands": hands, "objects": objects}}
            _write_json(frame_dir / "training_data.json", data)
            manifest["accepted_frames"] += 1
        manifest["status"] = "complete" if manifest["accepted_frames"] else "no_valid_frames"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        if hasattr(tracker, "close"):
            tracker.close()
        _write_json(output / "preprocess" / "manifest.json", manifest)
    print(f"Exported {manifest['accepted_frames']}/{len(frames)} measured frames to {all_data}")
    return 0 if manifest["accepted_frames"] else 2


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Recording directory with aligned RGB, metric depth and intrinsics")
    parser.add_argument("--output", required=True, help="New session directory, e.g. data/serve_bread/demos/session_002")
    parser.add_argument("--config", help="Optional YAML defaults; explicit CLI flags override its settings")
    parser.add_argument("--labels-file", help="Explicit stem -> finish flag JSON (default: INPUT/labels.json)")
    parser.add_argument("--depth-scale", type=float, help="Metres per PNG depth unit; NPY depth always contains metres")
    parser.add_argument("--hand-backend", choices=("mediapipe", "hamer", "json"), default="mediapipe")
    parser.add_argument("--hand-model", help="Local MediaPipe hand_landmarker.task (required for mediapipe/hamer)")
    parser.add_argument("--hamer-checkpoint", help="Local HaMeR checkpoint; installed package/config and MANO assets required")
    parser.add_argument("--keypoints-dir", help="Per-frame 21 RGB pixel keypoints (default: INPUT/hands)")
    parser.add_argument("--flip-handedness", action="store_true", help="Swap detector handedness for mirrored recordings")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--hand-origin", choices=("pinch", "palm"), default="pinch")
    parser.add_argument("--object-backend", choices=("dinosam", "masks"), default="dinosam")
    parser.add_argument("--masks-dir", help="Annotated per-frame masks (default: INPUT/masks)")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny", help="Cached identifier or local model directory")
    parser.add_argument("--sam2-checkpoint", help="Local SAM2 checkpoint; otherwise use HF cache")
    parser.add_argument("--sam2-config", default="configs/sam2/sam2_hiera_t.yaml")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--inpaint", choices=("lama", "telea", "none"), default="lama",
                        help="Algorithm is explicit and recorded; none retains human pixels")
    parser.add_argument("--lama-model", help="Local two-input LaMa ONNX model")
    parser.add_argument("--allow-download", action="store_true", help="Permit selected DINO/SAM2/LaMa model downloads; no implicit downloads")
    parser.add_argument("--device", default="cpu", help="Torch device; CPU default, CUDA only when explicitly selected")
    parser.add_argument("--smoothing-alpha", type=float, default=0.5, help="Causal EMA alpha in (0,1]; 1 disables smoothing")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--mask-dilation", type=int, default=4, help="Arm mask dilation radius in pixels")
    parser.add_argument("--motion-threshold", type=float, default=0.02, help="Object speed threshold for is_dynamic, in metres/second")
    return parser


def main(argv=None):
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_path = config_parser.parse_known_args(argv)[0].config
    if config_path:
        import yaml
        settings = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(settings, dict):
            raise ValueError("Preprocessing config must be a YAML mapping")
        metadata_settings = {"task": "serve_bread", "camera_frame": "fixed_camera", "depth_units": "metres"}
        for key, expected in metadata_settings.items():
            if key in settings and settings.pop(key) != expected:
                raise ValueError(f"Config {key} must be {expected}")
        allowed = {action.dest for action in parser._actions} - {"help", "input", "output", "config"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown preprocessing config keys: {sorted(unknown)}")
        for action in parser._actions:
            if action.dest in settings and action.choices and settings[action.dest] not in action.choices:
                raise ValueError(f"Invalid config value for {action.dest}: {settings[action.dest]}")
        parser.set_defaults(**settings)
    return prepare(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
