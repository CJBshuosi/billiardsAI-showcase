"""Real classic-OpenCV inference and a synthetic-only regression runner."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np

_PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    "white": (242, 242, 242),
    "yellow": (20, 210, 235),
    "blue": (205, 80, 35),
    "red": (35, 35, 220),
    "purple": (155, 55, 145),
    "orange": (25, 130, 240),
    "green": (35, 105, 25),
    "maroon": (55, 45, 125),
    "black": (20, 20, 20),
}

_COLOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "white": ("cue",),
    "yellow": ("1", "9"),
    "blue": ("2", "10"),
    "red": ("3", "11"),
    "purple": ("4", "12"),
    "orange": ("5", "13"),
    "green": ("6", "14"),
    "maroon": ("7", "15"),
    "black": ("8",),
}


class _ClassicBackend(Protocol):
    version: str

    def detect(self, image: object) -> tuple[Mapping[str, Any], ...]: ...


class ClassicDetectorBackend:
    """Detect circular foreground contours and classify their dominant colour."""

    version = "opencv-ball-locator/1.0"

    def __init__(
        self,
        *,
        minimum_radius_px: float = 10.0,
        maximum_radius_px: float = 40.0,
        foreground_distance: float = 28.0,
        minimum_circularity: float = 0.82,
        confidence_abstain_below: float = 0.50,
    ) -> None:
        self.minimum_radius_px = minimum_radius_px
        self.maximum_radius_px = maximum_radius_px
        self.foreground_distance = foreground_distance
        self.minimum_circularity = minimum_circularity
        self.confidence_abstain_below = confidence_abstain_below

    @staticmethod
    def _background_bgr(image_bgr: np.ndarray) -> np.ndarray:
        sample = image_bgr[::16, ::16].reshape(-1, 3)
        return np.median(sample, axis=0).astype(np.float32)

    @staticmethod
    def _classify_color(image_bgr: np.ndarray, contour: np.ndarray) -> str:
        contour_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        median = np.median(image_bgr[contour_mask == 255], axis=0).astype(np.float32)
        value = float(np.max(median))
        if value < 52.0:
            return "black"
        if float(np.max(median) - np.min(median)) < 24.0 and value > 150.0:
            return "white"
        scale = max(value / 220.0, 0.55)
        corrected = np.clip(median / scale, 0.0, 255.0)
        return min(
            _PALETTE_BGR,
            key=lambda color: float(
                np.linalg.norm(corrected - np.asarray(_PALETTE_BGR[color], dtype=np.float32))
            ),
        )

    def detect(self, image: object) -> tuple[dict[str, Any], ...]:
        image_bgr = np.asarray(image)
        background = self._background_bgr(image_bgr)
        distance = np.linalg.norm(image_bgr.astype(np.float32) - background, axis=2)
        foreground = np.where(distance >= self.foreground_distance, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            foreground,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: list[dict[str, Any]] = []
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

            local_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(local_mask, [contour], -1, 255, thickness=-1)
            contrast = min(1.0, float(np.mean(distance[local_mask == 255])) / 100.0)
            confidence = float(np.clip(0.72 * circularity + 0.28 * contrast, 0.0, 1.0))
            rejection_reason = None
            if circularity < self.minimum_circularity:
                rejection_reason = "occluded"
            elif confidence < self.confidence_abstain_below:
                rejection_reason = "low_confidence"

            detections.append(
                {
                    "center_px": (float(x_px), float(y_px)),
                    "radius_px": float(radius_px),
                    "circularity": circularity,
                    "dominant_color": color,
                    "confidence": confidence,
                    "candidates": _COLOR_CANDIDATES[color],
                    "mask_polygon_px": tuple(
                        (float(point[0][0]), float(point[0][1])) for point in contour
                    ),
                    "rejection_reason": rejection_reason,
                }
            )
        return tuple(sorted(detections, key=lambda item: item["center_px"]))


class ClassicCvAdapter:
    """Normalize real backend output; synthetic truth never enters this path."""

    def __init__(self, backend: _ClassicBackend) -> None:
        self._backend = backend

    def detect(self, frames: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        observations: list[dict[str, Any]] = []
        for frame in frames:
            for detection in self._backend.detect(frame["image"]):
                rejection = detection.get("rejection_reason")
                observation = {
                    "frame_id": frame["frame_id"],
                    "timestamp_ms": frame["timestamp_ms"],
                    "center_px": list(detection["center_px"]),
                    "radius_px": detection["radius_px"],
                    "confidence": detection["confidence"],
                    "identity_candidates": list(detection["candidates"]),
                    "supported": rejection is None,
                    "abstention_reason": rejection,
                    "detector_version": self._backend.version,
                }
                if "circularity" in detection:
                    observation["circularity"] = detection["circularity"]
                if "dominant_color" in detection:
                    observation["dominant_color"] = detection["dominant_color"]
                if "mask_polygon_px" in detection:
                    observation["mask"] = {
                        "format": "polygon_px",
                        "points": [list(point) for point in detection["mask_polygon_px"]],
                    }
                observations.append(observation)
        return tuple(observations)


def _table_image(*, width: int = 600, height: int = 400) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = (72, 132, 58)
    return image


def _decode_png(image: np.ndarray) -> tuple[np.ndarray, str]:
    encoded, png = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("failed to encode synthetic PNG")
    decoded = cv2.imdecode(png, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("failed to decode synthetic PNG")
    return decoded, hashlib.sha256(png.tobytes()).hexdigest()


def _run_case(
    *,
    case_id: str,
    frame_id: int,
    image: np.ndarray,
    adapter: ClassicCvAdapter,
) -> dict[str, Any]:
    decoded, image_sha256 = _decode_png(image)
    predictions = adapter.detect(
        ({"frame_id": frame_id, "timestamp_ms": frame_id * 40, "image": decoded},)
    )
    return {
        "case_id": case_id,
        "image": {
            "encoding": "png",
            "sha256": image_sha256,
            "width_px": int(decoded.shape[1]),
            "height_px": int(decoded.shape[0]),
        },
        "predictions": list(predictions),
    }


def run_synthetic_classic_cv_baseline() -> dict[str, Any]:
    """Run deterministic decoded-PNG cases through real pixel inference."""

    backend = ClassicDetectorBackend()
    adapter = ClassicCvAdapter(backend)

    nominal = _table_image()
    expected_centers = ((150.0, 200.0), (300.0, 200.0), (450.0, 200.0))
    for center, color in zip(
        expected_centers,
        ((242, 242, 242), (20, 210, 235), (205, 80, 35)),
        strict=True,
    ):
        cv2.circle(
            nominal,
            (int(center[0]), int(center[1])),
            22,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    nominal_case = _run_case(case_id="nominal-colours", frame_id=0, image=nominal, adapter=adapter)
    ordered_predictions = sorted(nominal_case["predictions"], key=lambda item: item["center_px"])
    center_errors = [
        float(
            np.hypot(
                prediction["center_px"][0] - expected[0],
                prediction["center_px"][1] - expected[1],
            )
        )
        for prediction, expected in zip(ordered_predictions, expected_centers, strict=False)
    ]
    matched_count = sum(error <= 2.0 for error in center_errors)
    nominal_case["metrics"] = {
        "expected_ball_count": 3,
        "matched_ball_count": matched_count,
        "recall": matched_count / 3,
        "maximum_center_error_px": max(center_errors, default=None),
    }

    blank_case = _run_case(
        case_id="blank-table",
        frame_id=1,
        image=_table_image(),
        adapter=adapter,
    )

    radius_filter = _table_image()
    cv2.circle(radius_filter, (150, 200), 5, (242, 242, 242), thickness=-1)
    cv2.circle(radius_filter, (450, 200), 50, (242, 242, 242), thickness=-1)
    radius_case = _run_case(
        case_id="radius-filter",
        frame_id=2,
        image=radius_filter,
        adapter=adapter,
    )
    radius_case["metrics"] = {"expected_filtered_blob_count": 2}

    occluded = _table_image()
    cv2.circle(occluded, (300, 200), 22, (205, 80, 35), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(occluded, (300, 175), (330, 225), (72, 132, 58), thickness=-1)
    occluded_case = _run_case(
        case_id="occluded-abstain",
        frame_id=3,
        image=occluded,
        adapter=adapter,
    )

    weak_only = _table_image()
    cv2.circle(weak_only, (150, 200), 22, (90, 155, 70), thickness=-1, lineType=cv2.LINE_AA)
    weak_decoded, weak_sha256 = _decode_png(weak_only)
    weak_confidence = backend.detect(weak_decoded)[0]["confidence"]
    with_strong = weak_only.copy()
    cv2.circle(with_strong, (450, 200), 22, (242, 242, 242), thickness=-1, lineType=cv2.LINE_AA)
    confidence_case = _run_case(
        case_id="local-confidence",
        frame_id=4,
        image=with_strong,
        adapter=adapter,
    )
    alongside_confidence = next(
        prediction["confidence"]
        for prediction in confidence_case["predictions"]
        if prediction["center_px"][0] < 300
    )
    confidence_case["comparison_image_sha256"] = weak_sha256
    confidence_case["metrics"] = {
        "weak_confidence_alone": weak_confidence,
        "weak_confidence_alongside_strong": alongside_confidence,
        "weak_confidence_delta": abs(alongside_confidence - weak_confidence),
    }

    cases = [nominal_case, blank_case, radius_case, occluded_case, confidence_case]
    predictions = [prediction for case in cases for prediction in case["predictions"]]
    supported_count = sum(prediction["supported"] for prediction in predictions)
    abstained_count = len(predictions) - supported_count
    passed = (
        nominal_case["metrics"]["recall"] == 1.0
        and not blank_case["predictions"]
        and not radius_case["predictions"]
        and len(occluded_case["predictions"]) == 1
        and occluded_case["predictions"][0]["abstention_reason"] == "occluded"
        and confidence_case["metrics"]["weak_confidence_delta"] <= 1e-12
    )
    return {
        "schema_version": "1.0",
        "status": "PASS_SYNTHETIC_REGRESSION_ONLY" if passed else "FAIL_SYNTHETIC_REGRESSION",
        "candidate_id": "classic-cv-local-contrast-v1",
        "input": {"synthetic_only": True, "real_video_count": 0},
        "runtime": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "implementation": {
            "source_ref": "codex/6-7-cloudbase-real-loop",
            "source_commit_inspected": "2403420",
            "detector_version": backend.version,
            "synthetic_truth_adapter": False,
            "inference_stages": [
                "foreground_distance",
                "morphology_open_close",
                "external_contours",
                "radius_and_circularity",
                "dominant_colour_candidates",
                "contour_local_confidence",
            ],
        },
        "summary": {
            "case_count": len(cases),
            "supported_prediction_count": supported_count,
            "abstained_prediction_count": abstained_count,
            "filtered_failure_count": 2,
        },
        "cases": cases,
        "interpretation": {
            "proves": [
                "decoded synthetic PNG pixel inference",
                "deterministic contour regression",
                "explicit abstention evidence",
            ],
            "does_not_prove": [
                "real-video accuracy",
                "tracking or event accuracy",
                "realtime performance",
            ],
        },
    }
