"""Table-masked frame motion and conservative shot candidate segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from pydantic import Field, field_validator, model_validator

from billiards_ai.lab.contracts import FrozenContract


@dataclass(frozen=True)
class MotionInputFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray


class MotionConfig(FrozenContract):
    pixel_diff_threshold: int = Field(default=12, ge=0, le=255)
    min_component_area: int = Field(default=4, ge=1)
    max_component_area: int = Field(default=1200, ge=1)
    blur_kernel: int = Field(default=3, ge=1)
    start_threshold: float = Field(default=6.0, ge=0)
    stop_threshold: float = Field(default=2.0, ge=0)
    min_shot_frames: int = Field(default=2, ge=1)

    @field_validator("blur_kernel")
    @classmethod
    def blur_kernel_is_odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("blur kernel must be odd")
        return value

    @model_validator(mode="after")
    def thresholds_have_hysteresis(self) -> MotionConfig:
        if self.start_threshold <= self.stop_threshold:
            raise ValueError("start threshold must exceed stop threshold")
        if self.max_component_area < self.min_component_area:
            raise ValueError("maximum component area must not be below minimum")
        return self


class MotionFrame(FrozenContract):
    frame_index: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0)
    energy: float = Field(ge=0)
    active_pixel_ratio: float = Field(ge=0, le=1)
    centroid_x: float | None = None
    centroid_y: float | None = None
    confidence: float = Field(ge=0, le=1)


class ShotCandidate(FrozenContract):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    start_ms: float = Field(ge=0)
    end_ms: float = Field(ge=0)
    peak_energy: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    review_status: Literal["unreviewed"] = "unreviewed"

    @model_validator(mode="after")
    def interval_is_non_empty(self) -> ShotCandidate:
        if self.end_frame <= self.start_frame or self.end_ms < self.start_ms:
            raise ValueError("shot candidate interval must be non-empty")
        return self


def _gray(image: np.ndarray, blur_kernel: int) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("motion frame must be grayscale or BGR")
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return gray


def motion_series(
    frames: Sequence[MotionInputFrame],
    table_mask: np.ndarray,
    config: MotionConfig | None = None,
) -> tuple[MotionFrame, ...]:
    """Measure frame differences strictly inside a fixed table mask."""

    if not frames:
        return ()
    config = config or MotionConfig()
    first_shape = frames[0].image.shape[:2]
    if table_mask.ndim != 2 or table_mask.shape != first_shape:
        raise ValueError("table mask shape must match video frames")
    if np.count_nonzero(table_mask) == 0:
        raise ValueError("table mask must contain active pixels")
    for previous, current in zip(frames, frames[1:], strict=False):
        if current.frame_index <= previous.frame_index:
            raise ValueError("motion frame indices must be strictly increasing")
        if current.timestamp_ms < previous.timestamp_ms:
            raise ValueError("motion frame timestamps must be monotonic")
        if current.image.shape[:2] != first_shape:
            raise ValueError("motion frames must share one image shape")

    mask = table_mask.astype(bool)
    mask_pixels = int(np.count_nonzero(mask))
    previous_gray = _gray(frames[0].image, config.blur_kernel)
    output = [
        MotionFrame(
            frame_index=frames[0].frame_index,
            timestamp_ms=frames[0].timestamp_ms,
            energy=0.0,
            active_pixel_ratio=0.0,
            centroid_x=None,
            centroid_y=None,
            confidence=0.0,
        )
    ]
    for frame in frames[1:]:
        current_gray = _gray(frame.image, config.blur_kernel)
        difference = cv2.absdiff(previous_gray, current_gray)
        thresholded = np.where(
            (difference > config.pixel_diff_threshold) & mask,
            np.uint8(255),
            np.uint8(0),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, 8)
        retained = np.zeros(first_shape, dtype=bool)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if config.min_component_area <= area <= config.max_component_area:
                retained |= labels == label
        active_pixels = int(np.count_nonzero(retained))
        if active_pixels:
            ys, xs = np.nonzero(retained)
            energy = float(difference[retained].sum(dtype=np.float64) / mask_pixels)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
        else:
            energy = 0.0
            centroid_x = None
            centroid_y = None
        confidence = min(1.0, energy / config.start_threshold) if config.start_threshold else 1.0
        output.append(
            MotionFrame(
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
                energy=energy,
                active_pixel_ratio=active_pixels / mask_pixels,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                confidence=confidence,
            )
        )
        previous_gray = current_gray
    return tuple(output)


def _end_timestamp(series: Sequence[MotionFrame], index: int) -> float:
    if index + 1 < len(series):
        return series[index + 1].timestamp_ms
    if len(series) > 1:
        step = series[-1].timestamp_ms - series[-2].timestamp_ms
        return series[-1].timestamp_ms + max(0.0, step)
    return series[index].timestamp_ms


def segment_shots(
    series: Sequence[MotionFrame],
    config: MotionConfig | None = None,
) -> tuple[ShotCandidate, ...]:
    """Convert hysteretic motion intervals into unreviewed shot candidates only."""

    if not series:
        return ()
    config = config or MotionConfig()
    candidates: list[ShotCandidate] = []
    start_index: int | None = None
    peak_energy = 0.0
    for index, frame in enumerate(series):
        if start_index is None:
            if frame.energy >= config.start_threshold:
                start_index = index
                peak_energy = frame.energy
            continue
        peak_energy = max(peak_energy, frame.energy)
        if frame.energy <= config.stop_threshold:
            start = series[start_index]
            end_frame = frame.frame_index + 1
            if end_frame - start.frame_index >= config.min_shot_frames:
                candidates.append(
                    ShotCandidate(
                        start_frame=start.frame_index,
                        end_frame=end_frame,
                        start_ms=start.timestamp_ms,
                        end_ms=_end_timestamp(series, index),
                        peak_energy=peak_energy,
                        confidence=min(1.0, peak_energy / config.start_threshold),
                    )
                )
            start_index = None
            peak_energy = 0.0
    if start_index is not None:
        start = series[start_index]
        end_frame = series[-1].frame_index + 1
        if end_frame - start.frame_index >= config.min_shot_frames:
            candidates.append(
                ShotCandidate(
                    start_frame=start.frame_index,
                    end_frame=end_frame,
                    start_ms=start.timestamp_ms,
                    end_ms=_end_timestamp(series, len(series) - 1),
                    peak_energy=peak_energy,
                    confidence=min(1.0, peak_energy / config.start_threshold),
                )
            )
    return tuple(candidates)
