"""Optional model adapters. Imports and downloads happen only when selected."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


class JsonHandTracker:
    """Read {right/left: {keypoints: [[u,v] x21], confidence: 1.0}} per frame.

    Pixels must follow standard 21-landmark order. Missing frames are tracking
    loss; model-produced 3D coordinates are never treated as measured depth.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def detect(self, image, stem, timestamp_ms):
        path = self.directory / f"{stem}.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected a hand dictionary: {path}")
        hands = {}
        for side in ("right", "left"):
            if side in data:
                entry = data[side]
                points = np.asarray(entry["keypoints"], dtype=float)
                if points.shape != (21, 2):
                    raise ValueError(f"Expected 21 RGB pixel keypoints for {side}: {path}")
                hands[side] = {"keypoints": points, "confidence": float(entry.get("confidence", 1.0))}
        return hands


class MediaPipeHandTracker:
    """MediaPipe Tasks VIDEO tracking using a user-provided .task model."""

    def __init__(self, model_path, *, flip_handedness=False):
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError("--hand-model must point to an installed hand_landmarker.task")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("MediaPipe tracking requires the optional mediapipe package") from exc
        self.mp, self.flip_handedness = mp, flip_handedness
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def detect(self, image, stem, timestamp_ms):
        timestamp_ms = max(int(timestamp_ms), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        result = self.tracker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)), timestamp_ms)
        h, w = image.shape[:2]
        hands = {}
        for landmarks, categories in zip(result.hand_landmarks, result.handedness):
            category = categories[0]
            side = category.category_name.lower()
            if self.flip_handedness:
                side = "left" if side == "right" else "right"
            if side not in {"left", "right"}:
                continue
            confidence = float(category.score)
            if side in hands and confidence <= hands[side]["confidence"]:
                continue
            hands[side] = {"keypoints": np.array([[p.x * w, p.y * h] for p in landmarks]),
                           "confidence": confidence}
        return hands

    def close(self):
        self.tracker.close()


class HaMeRHandTracker:
    """Refine MediaPipe hand crops with installed HaMeR; lift its 2D joints by RGB-D.

    The installed HaMeR package needs its YAML model config and licensed MANO
    assets in addition to --hamer-checkpoint. No checkpoint or MANO downloads
    are triggered here, and initialization errors propagate to the caller.
    """

    def __init__(self, detector, checkpoint, *, device="cpu"):
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("--hamer-checkpoint must point to an installed HaMeR checkpoint")
        try:
            import torch
            from hamer.models import load_hamer
        except ImportError as exc:
            raise ImportError("HaMeR requires an installed upstream hamer package and its dependencies") from exc
        self.detector, self.torch, self.device = detector, torch, device
        self.model, self.cfg = load_hamer(str(checkpoint))
        self.model = self.model.to(device).eval()

    def detect(self, image, stem, timestamp_ms):
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        detections = self.detector.detect(image, stem, timestamp_ms)
        h, w = image.shape[:2]
        refined = {}
        for side, entry in detections.items():
            points = entry["keypoints"]
            low, high = points.min(axis=0), points.max(axis=0)
            pad = (high - low) * 0.3
            low, high = np.maximum(low - pad, 0), np.minimum(high + pad, [w - 1, h - 1])
            if min(high - low) < 10:
                continue
            dataset = ViTDetDataset(self.cfg, img_cv2=image[:, :, ::-1].copy(),
                                    boxes=np.array([[*low, *high]], np.float32),
                                    right=np.array([side == "right"], np.float32))
            loader = self.torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
            batch = recursive_to(next(iter(loader)), self.device)
            with self.torch.inference_mode():
                output = self.model(batch)
                camera = output["pred_cam"].clone()
                camera[:, 1] *= 2 * batch["right"] - 1
                focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * batch["img_size"].max()
                translation = cam_crop_to_full(camera, batch["box_center"].float(),
                                                batch["box_size"].float(), batch["img_size"].float(), focal)
                joints = output["pred_keypoints_3d"][0].detach().cpu().numpy().copy()
                if side == "left":
                    joints[:, 0] *= -1
                joints += translation[0].detach().cpu().numpy()
            if joints.shape != (21, 3) or not np.isfinite(joints).all() or (joints[:, 2] <= 0).any():
                continue
            pixels = joints[:, :2] / joints[:, 2, None] * float(focal) + np.array([w / 2, h / 2])
            refined[side] = {"keypoints": pixels, "confidence": entry["confidence"]}
        return refined

    def close(self):
        self.detector.close()


class MaskObjectSegmenter:
    """Read user-annotated masks/<frame>/{obj1,obj2,arm}.png without models."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def segment(self, image, stem):
        masks = {}
        for name in ("obj1", "obj2", "arm"):
            path = self.directory / stem / f"{name}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing annotated segmentation mask: {path}")
            mask = np.asarray(Image.open(path).convert("L")) > 0
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask is not aligned to RGB: {path}")
            masks[name] = mask
        return masks


class GroundedSamSegmenter:
    """Grounding DINO text boxes + SAM2 masks; inference defaults to CPU."""

    def __init__(self, *, dino_model="IDEA-Research/grounding-dino-tiny", sam_checkpoint=None,
                 sam_config="configs/sam2/sam2_hiera_t.yaml", allow_download=False,
                 sam_repo="facebook/sam2-hiera-tiny", sam_file="sam2_hiera_tiny.pt",
                 device="cpu", box_threshold=0.3):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("DINO/SAM2 requires optional torch, transformers, huggingface-hub and upstream SAM2") from exc
        if sam_checkpoint is None:
            from huggingface_hub import hf_hub_download
            sam_checkpoint = hf_hub_download(sam_repo, filename=sam_file, local_files_only=not allow_download)
        if not Path(sam_checkpoint).is_file():
            raise FileNotFoundError(f"SAM2 checkpoint is missing: {sam_checkpoint}")
        self.torch, self.device, self.box_threshold = torch, device, box_threshold
        self.processor = AutoProcessor.from_pretrained(dino_model, local_files_only=not allow_download)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model, local_files_only=not allow_download).to(device).eval()
        self.predictor = SAM2ImagePredictor(build_sam2(sam_config, str(sam_checkpoint), device=device))

    def segment(self, image, stem):
        masks = {}
        h, w = image.shape[:2]
        with self.torch.inference_mode():
            self.predictor.set_image(image)
            for name, text in {"obj1": "bread .", "obj2": "plate .",
                               "arm": "human arms . human hands . person ."}.items():
                inputs = self.processor(images=Image.fromarray(image), text=text, return_tensors="pt").to(self.device)
                output = self.detector(**inputs)
                scores = output.logits.sigmoid()[0].amax(-1)
                indices = (scores > self.box_threshold).nonzero().flatten()
                if name != "arm" and len(indices):
                    indices = indices[scores[indices].argmax()].reshape(1)
                combined = np.zeros((h, w), bool)
                for index in indices:
                    cx, cy, bw, bh = output.pred_boxes[0, index].detach().cpu().numpy() * [w, h, w, h]
                    box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
                    predicted, _, _ = self.predictor.predict(box=box, multimask_output=False)
                    combined |= np.asarray(predicted).reshape(-1, h, w).any(axis=0)
                masks[name] = combined
        return masks


class Inpainter:
    """Explicit LaMa, OpenCV Telea, or no inpainting; no silent substitutions."""

    def __init__(self, algorithm="lama", model_path=None, *, allow_download=False, input_size=512):
        self.algorithm, self.input_size, self.session = algorithm, input_size, None
        if algorithm not in {"lama", "telea", "none"}:
            raise ValueError("Choose lama, telea, or none for inpainting")
        if algorithm == "lama":
            if model_path is None and allow_download:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download("Carve/LaMa-ONNX", filename="lama_fp32.onnx")
            if not model_path or not Path(model_path).is_file():
                raise FileNotFoundError("LaMa requires --lama-model /path/model.onnx or --allow-download")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError("LaMa requires the optional onnxruntime package") from exc
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            if len(self.session.get_inputs()) != 2:
                raise ValueError("Expected a LaMa ONNX model with image and mask inputs")
            self.model_path = str(model_path)
        elif algorithm == "telea":
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("Telea inpainting requires optional opencv-python") from exc
            self.cv2 = cv2

    def inpaint(self, image, mask):
        mask = np.asarray(mask, bool)
        if mask.shape != image.shape[:2]:
            raise ValueError("Inpainting mask is not aligned to RGB")
        if self.algorithm == "none" or not mask.any():
            return image.copy()
        if self.algorithm == "telea":
            return self.cv2.inpaint(image, mask.astype(np.uint8) * 255, 3, self.cv2.INPAINT_TELEA)
        h, w = image.shape[:2]
        size = self.input_size
        rgb = np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR), np.float32)
        binary = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (size, size), Image.Resampling.NEAREST), np.float32) / 255
        inputs = self.session.get_inputs()
        feeds = {inputs[0].name: rgb.transpose(2, 0, 1)[None] / 255,
                 inputs[1].name: binary[None, None]}
        result = np.asarray(self.session.run(None, feeds)[0])
        if result.ndim != 4 or result.shape[0] != 1 or result.shape[1] != 3:
            raise ValueError("Expected LaMa NCHW RGB output")
        result = result[0].transpose(1, 2, 0)
        if not np.isfinite(result).all():
            raise ValueError("LaMa produced non-finite output")
        result = np.clip(result * 255 if result.max() <= 1.1 else result, 0, 255).astype(np.uint8)
        result = np.asarray(Image.fromarray(result).resize((w, h), Image.Resampling.BILINEAR))
        # Preserve unmasked pixels exactly; only feather the inside of the mask.
        alpha = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
            ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
        alpha = (alpha * mask)[..., None]
        return np.clip(result * alpha + image * (1 - alpha), 0, 255).astype(np.uint8)
