"""Independent offline artwork; snapshots in renderer millimetres become BGR pixels.

This helper owns no motion, physics, rule truth, product events or model feedback.
Renderer y points down. Artwork is deterministic stylization, not model validation.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil, floor, isfinite
from numbers import Integral, Real

import cv2
import numpy as np

# Fixed standard-ish BGR swatches, chosen without model output.
_HUES = (
    (0, 210, 255),  # yellow
    (200, 70, 25),  # blue
    (35, 35, 220),  # red
    (140, 45, 110),  # purple
    (0, 125, 245),  # orange
    (65, 140, 25),  # green
    (45, 35, 110),  # maroon
)
_SCALE = 1400 / 2540
_RADIUS_PX = 28.575 * _SCALE


@dataclass(frozen=True, slots=True)
class BallVisual:
    """Offline snapshot value. Coordinates are renderer x/right and y/down, in mm."""

    number: int
    x_mm: float
    y_mm: float
    visible: bool = True


def _number(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("ball number must be an integer, excluding bool")
    if not 0 <= value <= 15:
        raise ValueError("ball number must be in 0..15")
    return int(value)


def _finite(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("coordinates and radius must be real numbers, excluding bool")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("coordinates and radius must be finite") from error
    if not isfinite(result):
        raise ValueError("coordinates and radius must be finite")
    return result


def draw_ball(
    frame: np.ndarray, *, number: int, center_px: tuple[float, float], radius_px: float
) -> None:
    """Draw ordinary numbered artwork in place, clipped to the supplied circle/image.

    Validate all arguments before touching pixels. Finite off-image centres are legal;
    there is no rail clamp or homography. Numeral ink is clipped to its white disk.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be H x W x 3 uint8 BGR")
    if not frame.flags.c_contiguous or not frame.flags.writeable or 0 in frame.shape:
        raise ValueError("frame must be nonempty, writable and C-contiguous")
    number = _number(number)
    if not isinstance(center_px, (tuple, list)) or len(center_px) != 2:
        raise TypeError("center_px must contain two finite coordinates")
    cx, cy = (_finite(value) for value in center_px)
    radius = _finite(radius_px)
    if radius < 1:
        raise ValueError("radius_px must be at least 1")

    height, width = frame.shape[:2]
    # Clamp floating bounds before integer conversion, including very distant centres.
    x0, x1 = ceil(max(0, min(width, cx - radius))), floor(max(-1, min(width - 1, cx + radius)))
    y0, y1 = ceil(max(0, min(height, cy - radius))), floor(max(-1, min(height - 1, cy + radius)))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    dx, dy = (xx - cx) / radius, (yy - cy) / radius
    distance_squared = dx * dx + dy * dy
    ball = distance_squared <= 1
    artwork = np.full((*ball.shape, 3), 255, dtype=np.uint8)
    if number == 8:
        artwork[ball] = (0, 0, 0)
    elif number:
        hue = _HUES[(number - 1) % 8] if number < 8 else _HUES[number - 9]
        colored = ball if number < 8 else ball & (np.abs(dy) <= 0.45)
        artwork[colored] = hue
    if number:
        disk = distance_squared <= 0.43**2
        artwork[disk] = 255
        # A bounded texture keeps typography deterministic even for very large radii.
        ink = np.zeros((64, 64), dtype=np.uint8)
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (base_width, _), _ = cv2.getTextSize(text, font, 1, 2)
        font_scale = min(1.25, 48 / base_width)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
        cv2.putText(
            ink,
            text,
            ((64 - text_width) // 2, (64 + text_height) // 2),
            font,
            font_scale,
            255,
            2,
            cv2.LINE_8,
        )
        ix = np.clip(((dx[disk] / 0.43 + 1) * 31.5).round().astype(int), 0, 63)
        iy = np.clip(((dy[disk] / 0.43 + 1) * 31.5).round().astype(int), 0, 63)
        artwork[disk] = np.where(ink[iy, ix, None] != 0, 0, 255)
    roi = frame[y0 : y1 + 1, x0 : x1 + 1]
    roi[ball] = artwork[ball]


def render_frame(balls: Iterable[BallVisual]) -> np.ndarray:
    """Return a 1920x1080 snapshot; reject duplicate numbers, even if invisible.

    All values are validated before creating/drawing the snapshot. BallVisual is an
    offline input only; no snapshot metadata accompanies the returned pixel array.
    """
    snapshots = list(balls)
    seen = set()
    validated = []
    for ball in snapshots:
        if not isinstance(ball, BallVisual):
            raise TypeError("snapshots must contain BallVisual values")
        number = _number(ball.number)
        x, y = _finite(ball.x_mm), _finite(ball.y_mm)
        if not isinstance(ball.visible, bool):
            raise TypeError("visible must be bool")
        if number in seen:
            raise ValueError("duplicate ball number in snapshot")
        seen.add(number)
        validated.append((number, x, y, ball.visible))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[147:848, 240:1641] = (45, 110, 45)
    for number, x, y, visible in validated:
        if visible:
            draw_ball(
                frame,
                number=number,
                center_px=(240 + x * _SCALE, 147 + y * _SCALE),
                radius_px=_RADIUS_PX,
            )
    return frame
