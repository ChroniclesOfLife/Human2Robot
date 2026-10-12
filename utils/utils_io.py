import os
import json
import cv2
import numpy as np
from typing import Optional, Any

def read_json(path: str) -> Optional[dict]:
    if not path or (not os.path.exists(path)):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_imread_gray(path: str, h: int, w: int) -> np.ndarray:
    """Read grayscale image safely and resize."""
    if path and os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
            return im
    return np.zeros((h, w), dtype=np.uint8)


def safe_imread_rgb(path: str, h: int, w: int) -> np.ndarray:
    """Read RGB image safely and resize."""
    if path and os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            return im
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_json_serializable(obj: Any) -> Any:
    """Recursive helper to convert types/numpy to serializable JSON."""
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (list, tuple)): return [make_json_serializable(i) for i in obj]
    if isinstance(obj, dict): return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, type): return str(obj)
    if hasattr(obj, '__dict__'): return make_json_serializable(vars(obj))
    return str(obj)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
