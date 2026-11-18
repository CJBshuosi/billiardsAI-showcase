"""Interactive checkerboard sampling for a fixed Mainline 1 USB camera."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from billiards_ai.lab.mainline1_lens_calibration import LensCalibrationSample
from billiards_ai.lab.mainline1_usb_capture import CaptureFactory, open_usb_capture


class LensSampleCaptureError(RuntimeError):
    """The USB stream cannot produce a trustworthy lens sample set."""


@dataclass(frozen=True)
class CheckerboardSpec:
    columns: int
    rows: int
    square_size_mm: float

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2 or self.square_size_mm <= 0:
            raise ValueError("checkerboard geometry must be positive")


CheckerboardDetector = Callable[..., tuple[bool, Any]]
CheckerboardRoi = tuple[int, int, int, int]


def select_checkerboard_roi(frame: Any) -> CheckerboardRoi:
    """Select a search region on the original image, without resampling pixels."""
    name = "Select whole board + white margin; ENTER confirm, C cancel"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 720)
    try:
        return tuple(
            int(v) for v in cv2.selectROI(name, frame, showCrosshair=True, fromCenter=False)
        )
    finally:
        cv2.destroyWindow(name)


def detect_checkerboard_sample(
    frame: Any,
    spec: CheckerboardSpec,
    *,
    detector: CheckerboardDetector = cv2.findChessboardCornersSB,
    roi: CheckerboardRoi | None = None,
) -> LensCalibrationSample | None:
    """Return metric/image point pairs when every inner corner is visible."""

    x, y, width, height = 0, 0, frame.shape[1], frame.shape[0]
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise ValueError("ROI must contain four integer pixel coordinates")
        x, y, width, height = roi
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > frame.shape[1]
            or y + height > frame.shape[0]
        ):
            raise ValueError("ROI must be inside the original frame")
    gray = cv2.cvtColor(frame[y : y + height, x : x + width], cv2.COLOR_BGR2GRAY)
    found, corners = detector(
        gray,
        (spec.columns, spec.rows),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not found or corners is None:
        return None
    flattened = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    expected = spec.columns * spec.rows
    if len(flattened) != expected:
        return None
    if (
        not np.isfinite(flattened).all()
        or (flattened < 0).any()
        or (flattened[:, 0] >= width).any()
        or (flattened[:, 1] >= height).any()
    ):
        return None
    # Calibration always consumes original-frame pixels, not ROI-relative pixels.
    flattened = flattened + np.asarray([x, y], dtype=np.float32)
    object_points = tuple(
        (column * spec.square_size_mm, row * spec.square_size_mm, 0.0)
        for row in range(spec.rows)
        for column in range(spec.columns)
    )
    image_points = tuple((float(point[0]), float(point[1])) for point in flattened)
    return LensCalibrationSample(object_points=object_points, image_points=image_points)


def collect_usb_checkerboard_samples(
    *,
    runtime_index: int,
    spec: CheckerboardSpec,
    target_samples: int,
    width: int = 1920,
    height: int = 1080,
    fps: float = 60.0,
    capture_factory: CaptureFactory = open_usb_capture,
    detector: CheckerboardDetector = cv2.findChessboardCornersSB,
    key_reader: Callable[[int], int] = cv2.waitKey,
    presenter: Callable[[str, Any], None] = cv2.imshow,
    window_closer: Callable[[], None] = cv2.destroyAllWindows,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    cadence_sample_frames: int = 120,
    require_mjpg: bool = False,
    expected_controls: dict[int, float] | None = None,
    capture_evidence: dict[str, Any] | None = None,
    warmup_frames: int = 0,
    roi_selector: Callable[[Any], CheckerboardRoi] | None = None,
    evidence_directory: Path | None = None,
) -> tuple[LensCalibrationSample, ...]:
    """Collect manually accepted detected poses and fail closed on stream fallback."""

    if runtime_index < 0 or target_samples < 8 or cadence_sample_frames < 2 or warmup_frames < 0:
        raise ValueError(
            "runtime index, sample count, and cadence sample must satisfy the fixed gate"
        )
    if evidence_directory is not None:
        evidence_directory.mkdir(parents=True, exist_ok=False)
    capture = capture_factory(runtime_index)
    samples: list[LensCalibrationSample] = []
    evidence = capture_evidence if capture_evidence is not None else {}
    roi = None
    roi_attempt = 0
    initial_saved = False

    def save_frame(label: str, frame: Any, sample: LensCalibrationSample | None = None) -> None:
        if evidence_directory is None:
            return
        annotated = frame.copy()
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 2)
        if sample is not None:
            points = np.asarray(sample.image_points, dtype=np.float32).reshape(-1, 1, 2)
            cv2.drawChessboardCorners(annotated, (spec.columns, spec.rows), points, True)
        metadata = {
            "label": label,
            "roi": list(roi) if roi is not None else None,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "image_points": sample.image_points if sample is not None else None,
            "coordinate_space": "original_full_frame_pixels",
            "files": {},
        }
        for suffix, pixels in ((".png", frame), ("-corners.png", annotated)):
            ok, encoded = cv2.imencode(".png", pixels)
            if not ok:
                raise LensSampleCaptureError("could not encode frame evidence")
            name = label + suffix
            data = encoded.tobytes()
            with (evidence_directory / name).open("xb") as handle:
                handle.write(data)
            metadata["files"][name] = hashlib.sha256(data).hexdigest()
        with (evidence_directory / (label + ".json")).open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        evidence.setdefault("frame_evidence", []).append(metadata)

    def validate_readback() -> dict[str, float]:
        if require_mjpg and round(capture.get(cv2.CAP_PROP_FOURCC)) != cv2.VideoWriter_fourcc(
            *"MJPG"
        ):
            raise LensSampleCaptureError("camera did not preserve MJPG capture mode")
        controls = {str(prop): float(capture.get(prop)) for prop in (expected_controls or {})}
        if any(controls[str(prop)] != value for prop, value in (expected_controls or {}).items()):
            raise LensSampleCaptureError(f"camera control readback changed: {controls}")
        return controls

    try:
        if not capture.isOpened():
            raise LensSampleCaptureError("camera could not be opened")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        capture.set(cv2.CAP_PROP_FPS, fps)
        # DirectShow's FPS negotiation can reset the media subtype.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for _ in range(warmup_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LensSampleCaptureError("camera frame read failed during warmup")
            if tuple(frame.shape[:2]) != (height, width):
                raise LensSampleCaptureError("delivered frame geometry did not match fixed mode")
        evidence["warmup_frames"] = warmup_frames
        observed_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        observed_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        observed_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if observed_width != width or observed_height != height or abs(observed_fps - fps) > 0.5:
            raise LensSampleCaptureError(
                f"camera did not preserve exact {width}x{height}@{fps:g} capture mode"
            )
        evidence.update(
            width=observed_width,
            height=observed_height,
            fps=observed_fps,
            clock="perf_counter_ns",
            controls_before_capture=validate_readback(),
        )

        timestamps_ns: list[int] = []
        for _ in range(cadence_sample_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LensSampleCaptureError("camera frame read failed during cadence gate")
            if tuple(frame.shape[:2]) != (height, width):
                raise LensSampleCaptureError("delivered frame geometry did not match fixed mode")
            timestamps_ns.append(monotonic_ns())
        delivery_span_ns = timestamps_ns[-1] - timestamps_ns[0]
        if delivery_span_ns <= 0:
            raise LensSampleCaptureError("camera delivery cadence could not be measured")
        measured_fps = (len(timestamps_ns) - 1) / (delivery_span_ns / 1_000_000_000)
        evidence["measured_fps"] = measured_fps
        if abs(measured_fps - fps) > fps * 0.05:
            raise LensSampleCaptureError("measured frame delivery cadence did not match fixed mode")

        while len(samples) < target_samples:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise LensSampleCaptureError("camera frame read failed")
            if tuple(frame.shape[:2]) != (height, width):
                raise LensSampleCaptureError("delivered frame geometry did not match fixed mode")
            if not initial_saved:
                save_frame("initial", frame)
                initial_saved = True
            sample = detect_checkerboard_sample(frame, spec, detector=detector, roi=roi)
            preview = frame.copy()
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 2)
            if sample is not None:
                detected = np.asarray(sample.image_points, dtype=np.float32).reshape(-1, 1, 2)
                cv2.drawChessboardCorners(preview, (spec.columns, spec.rows), detected, True)
            status = (
                f"detected - SPACE to capture ({len(samples)}/{target_samples})"
                if sample is not None
                else f"show full checkerboard ({len(samples)}/{target_samples})"
            )
            if roi_selector is not None:
                status += " | ROI" if roi is not None else " | FULL"
                status += " | R: select board"
            cv2.putText(
                preview,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if sample is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            presenter("billiardsAI lens calibration - SPACE capture, Q cancel", preview)
            key = key_reader(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                raise LensSampleCaptureError("lens sample collection was cancelled")
            if key in (ord("r"), ord("R")) and roi_selector is not None:
                selected = roi_selector(frame)
                if selected[2] > 0 and selected[3] > 0:
                    roi = selected
                    roi_attempt += 1
                    save_frame(f"roi-{roi_attempt:03d}", frame)
                # R never accepts the frozen selection frame. Detect on the next read.
                continue
            if key == ord(" ") and sample is not None:
                validate_readback()
                save_frame(f"sample-{len(samples) + 1:03d}", frame, sample)
                samples.append(sample)
        evidence["controls_after_capture"] = validate_readback()
        return tuple(samples)
    finally:
        capture.release()
        window_closer()
