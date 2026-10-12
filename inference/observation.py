"""Read a portable recorded frame without camera SDK or capture-system imports."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inference.interfaces import ObjectState


@dataclass
class RecordedObservation:
    rgb_bgr: np.ndarray
    K: np.ndarray
    hands_in_cam: dict[str, np.ndarray]
    grippers: dict[str, float]
    objects: dict[str, ObjectState]
    anchor_key: str
    source: Path
    image_anchor: ObjectState


def _pose(value):
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Observation poses must be finite 4x4 matrices")
    return pose


def load_recorded_observation(frame: str | Path, image_name="observation.png") -> RecordedObservation:
    path = Path(frame).resolve()
    path = path / "training_data.json" if path.is_dir() else path
    record = json.loads(path.read_text(encoding="utf-8"))
    metadata = record["metadata"]
    # The camera-frame training target uses cam0, rather than the moving capture camera.
    camera_to_world = metadata.get("world_transforms", {}).get("cam0", metadata.get("c2w"))
    world_to_camera = np.linalg.inv(_pose(camera_to_world))
    entities = record["entities"]
    hands = entities.get("hands") or {}
    hands_in_cam = {side: world_to_camera @ _pose(item["T_hand_to_world"]) for side, item in hands.items()}
    grippers = {side: float(float(item.get("grasp", 0)) > 0.5) for side, item in hands.items()}
    objects = {}
    for key, item in entities.get("objects", {}).items():
        pose_world = _pose(item["T_obj_to_world"])
        points_world = np.asarray(record.get("obs", {}).get("objects_kpts", {}).get(key, {}).get("world", []), np.float32).reshape(-1, 3)
        local = (np.linalg.inv(pose_world)[:3, :3] @ (points_world - pose_world[:3, 3]).T).T
        objects[key] = ObjectState(world_to_camera @ pose_world, local)
    anchor_key = metadata.get("anchor_key", "obj1")
    if anchor_key not in objects or not hands_in_cam:
        raise ValueError("Frame needs an observed hand and its anchor object")
    # Resolve from the frame directory, never from paths embedded by the capture checkout.
    image_path = path.parent / Path(image_name).name
    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot read training observation image: {image_path}")
    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read training observation image: {image_path}")
    K = np.asarray(metadata["k"], dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("Observation intrinsics must be a finite 3x3 matrix")
    # Region attention projects into the captured image, whose camera can move
    # independently of the fixed training reference cam0.
    image_world_to_camera = np.linalg.inv(_pose(metadata.get("c2w", camera_to_world)))
    anchor_world = _pose(entities["objects"][anchor_key]["T_obj_to_world"])
    image_anchor = ObjectState(image_world_to_camera @ anchor_world)
    return RecordedObservation(rgb, K, hands_in_cam, grippers, objects, anchor_key, path.parent, image_anchor)


def first_recorded_frame(directory: str | Path) -> Path:
    for frame in sorted(Path(directory).iterdir()):
        if frame.is_dir() and frame.name.isdigit():
            try:
                load_recorded_observation(frame)
                return frame
            except (KeyError, ValueError, FileNotFoundError):
                continue
    raise FileNotFoundError(f"No valid recorded frame in {directory}")
