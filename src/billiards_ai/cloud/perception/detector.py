"""Deterministic classic-OpenCV ball observations for normalized frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from billiards_ai.shared.constants import CONFIDENCE_ABSTAIN_BELOW
from billiards_ai.shared.models.detection import (
    BallCandidate,
    BallColor,
    PixelObservation,
    RejectionReason,
)


@dataclass(frozen=True)
class PixelDetection:
    candidates: tuple[BallCandidate, ...]
    confidence: float
    raw_observation: PixelObservation
    rejection_reason: RejectionReason | None = None


_PALETTE_BGR: dict[BallColor, tuple[int, int, int]] = {
    BallColor.WHITE: (242, 242, 242),
    BallColor.YELLOW: (20, 210, 235),
    BallColor.BLUE: (205, 80, 35),
    BallColor.RED: (35, 35, 220),
    BallColor.PURPLE: (155, 55, 145),
    BallColor.ORANGE: (25, 130, 240),
    BallColor.GREEN: (35, 105, 25),
    BallColor.MAROON: (55, 45, 125),
    BallColor.BLACK: (20, 20, 20),
}

_COLOR_IDS: dict[BallColor, tuple[int, ...]] = {
    BallColor.WHITE: (0,),
    BallColor.YELLOW: (1, 9),
    BallColor.BLUE: (2, 10),
    BallColor.RED: (3, 11),
    BallColor.PURPLE: (4, 12),
    BallColor.ORANGE: (5, 13),
    BallColor.GREEN: (6, 14),
    BallColor.MAROON: (7, 15),
    BallColor.BLACK: (8,),
}


class ClassicCvBallDetector:
    """Find circular foreground blobs and classify their dominant billiard color."""

    version = "opencv-ball-locator/1.0"

    def __init__(
        self,
        *,
        minimum_radius_px: float = 10.0,
        maximum_radius_px: float = 40.0,
        foreground_distance: float = 28.0,
        minimum_circularity: float = 0.82,
    ):
        self.minimum_radius_px = minimum_radius_px
        self.maximum_radius_px = maximum_radius_px
        self.foreground_distance = foreground_distance
        self.minimum_circularity = minimum_circularity

    @staticmethod
    def _background_bgr(image_bgr: np.ndarray) -> np.ndarray:
        sample = image_bgr[::16, ::16].reshape(-1, 3)
        return np.median(sample, axis=0).astype(np.float32)

    @staticmethod
    def _classify_color(image_bgr: np.ndarray, contour: np.ndarray) -> BallColor:
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        pixels = image_bgr[mask == 255]
        median = np.median(pixels, axis=0).astype(np.float32)
        value = float(np.max(median))
        if value < 52.0:
            return BallColor.BLACK
        if float(np.max(median) - np.min(median)) < 24.0 and value > 150.0:
            return BallColor.WHITE
        scale = max(value / 220.0, 0.55)
        corrected = np.clip(median / scale, 0.0, 255.0)
        return min(
            _PALETTE_BGR,
            key=lambda color: float(
                np.linalg.norm(corrected - np.asarray(_PALETTE_BGR[color], dtype=np.float32))
            ),
        )

    @staticmethod
    def _candidates(color: BallColor) -> tuple[BallCandidate, ...]:
        ball_ids = _COLOR_IDS[color]
        probability = 1.0 / len(ball_ids)
        return tuple(
            BallCandidate(ball_id=ball_id, color=color, confidence=probability)
            for ball_id in ball_ids
        )

    def detect(self, image_bgr: np.ndarray) -> tuple[PixelDetection, ...]:
        background = self._background_bgr(image_bgr)
        distance = np.linalg.norm(image_bgr.astype(np.float32) - background, axis=2)
        mask = np.where(distance >= self.foreground_distance, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        observations: list[PixelDetection] = []
        minimum_area = np.pi * self.minimum_radius_px**2 * 0.35
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < minimum_area:
                continue
            (x_px, y_px), radius_px = cv2.minEnclosingCircle(contour)
            if radius_px < self.minimum_radius_px or radius_px > self.maximum_radius_px:
                continue
            perimeter = float(cv2.arcLength(contour, closed=True))
            circularity = 0.0 if perimeter <= 0.0 else 4.0 * np.pi * area / perimeter**2
            circularity = float(np.clip(circularity, 0.0, 1.0))
            color = self._classify_color(image_bgr, contour)
            candidates = self._candidates(color)
            # Confidence must describe this contour.  A bright, strong ball
            # elsewhere in the frame must not make a weak/occluded contour
            # appear reliable.
            local_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(local_mask, [contour], -1, 255, thickness=-1)
            contrast = min(1.0, float(np.mean(distance[local_mask == 255])) / 100.0)
            confidence = float(np.clip(0.72 * circularity + 0.28 * contrast, 0.0, 1.0))
            raw = PixelObservation(
                center_px=(float(x_px), float(y_px)),
                radius_px=float(radius_px),
                circularity=circularity,
                color=color,
            )
            rejection = None
            if circularity < self.minimum_circularity:
                rejection = RejectionReason.OCCLUDED
            elif confidence < CONFIDENCE_ABSTAIN_BELOW:
                rejection = RejectionReason.LOW_CONFIDENCE
            observations.append(
                PixelDetection(
                    candidates=candidates,
                    confidence=confidence,
                    raw_observation=raw,
                    rejection_reason=rejection,
                )
            )
        return tuple(
            sorted(
                observations,
                key=lambda item: item.raw_observation.center_px,
            )
        )
