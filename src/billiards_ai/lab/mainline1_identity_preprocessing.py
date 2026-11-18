"""Single image preprocessing implementation shared by runtime and training."""

from __future__ import annotations

import math

import cv2
import numpy as np


class IdentityCropExportError(RuntimeError):
    """The reviewed detector dataset cannot safely produce identity crops."""


def square_identity_crop(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    context_scale: float,
    output_size: int,
) -> np.ndarray:
    x, y, width, height = bbox
    side = max(2, int(math.ceil(max(width, height) * context_scale)))
    center_x = x + width / 2
    center_y = y + height / 2
    left = int(math.floor(center_x - side / 2))
    top = int(math.floor(center_y - side / 2))
    right = left + side
    bottom = top + side
    image_height, image_width = image.shape[:2]
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_width, right)
    source_bottom = min(image_height, bottom)
    if source_left >= source_right or source_top >= source_bottom:
        raise IdentityCropExportError("identity crop lies outside its source image")
    crop = image[source_top:source_bottom, source_left:source_right]
    crop = cv2.copyMakeBorder(
        crop,
        source_top - top,
        bottom - source_bottom,
        source_left - left,
        right - source_right,
        cv2.BORDER_REPLICATE,
    )
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)


def normalise_identity_features(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("DINOv2 features are invalid")
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.float32(1e-12))
