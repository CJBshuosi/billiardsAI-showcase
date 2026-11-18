"""Deterministic mask-color baseline inspired by DEADPOOL_Project.

RF-DETR/SAM2 remain responsible for instances. This backend only extracts
coarse cue/eight/solid/stripe evidence from pixels inside an existing mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import cv2
import numpy as np


def _unknown_scores() -> dict[str, dict[str, float]]:
    return {
        "exact_scores": {"unknown": 1.0, "1": 0.0},
        "group_scores": {"unknown": 1.0, "cue": 0.0, "eight": 0.0, "solid": 0.0, "stripe": 0.0},
        "visibility_scores": {"occluded": 1.0, "visible": 0.0},
    }


class DeadpoolColorBackend:
    version = "deadpool-mask-color/1"

    def predict(self, crop: object) -> Mapping[str, Mapping[str, float]]:
        payload = cast(Mapping[str, np.ndarray], crop)
        image = np.asarray(payload["image"])
        mask = np.asarray(payload["mask"])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR array")
        if mask.ndim != 2 or image.shape[:2] != mask.shape:
            raise ValueError("image and mask must have the same height and width")
        selected = mask.astype(bool)
        if int(selected.sum()) < 16:
            return _unknown_scores()

        pixels = image[selected]
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        white_ratio = float(np.mean((hsv[:, 1] < 35) & (hsv[:, 2] > 150)))
        black_ratio = float(np.mean(hsv[:, 2] < 55))
        color_ratio = float(np.mean((hsv[:, 1] >= 55) & (hsv[:, 2] >= 70)))

        if white_ratio >= 0.75:
            group_scores = {
                "cue": 0.95,
                "eight": 0.0,
                "solid": 0.01,
                "stripe": 0.02,
                "unknown": 0.02,
            }
            exact_scores = {"unknown": 0.85, "cue": 0.14, "1": 0.01}
        elif black_ratio >= 0.70:
            group_scores = {
                "cue": 0.0,
                "eight": 0.95,
                "solid": 0.01,
                "stripe": 0.01,
                "unknown": 0.03,
            }
            exact_scores = {"8": 0.90, "unknown": 0.09, "1": 0.01}
        elif 0.15 <= white_ratio <= 0.70 and color_ratio >= 0.15:
            group_scores = {
                "cue": 0.01,
                "eight": 0.01,
                "solid": 0.04,
                "stripe": 0.90,
                "unknown": 0.04,
            }
            exact_scores = {"unknown": 0.90, "9": 0.05, "10": 0.05}
        elif color_ratio >= 0.45 and white_ratio < 0.15:
            group_scores = {
                "cue": 0.01,
                "eight": 0.01,
                "solid": 0.90,
                "stripe": 0.04,
                "unknown": 0.04,
            }
            exact_scores = {"unknown": 0.90, "1": 0.05, "2": 0.05}
        else:
            return _unknown_scores()

        return {
            "exact_scores": exact_scores,
            "group_scores": group_scores,
            "visibility_scores": {"visible": 0.98, "occluded": 0.02},
        }
