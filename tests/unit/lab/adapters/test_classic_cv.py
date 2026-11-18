import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import billiards_ai.lab.adapters.classic_cv as classic_cv
from billiards_ai.lab.adapters.classic_cv import ClassicCvAdapter
from billiards_ai.lab.cli import main


class FakeClassicDetector:
    version = "fake-classic/1"

    def detect(self, image: object) -> tuple[dict, ...]:
        assert image == "synthetic-frame"
        return (
            {
                "center_px": (42.0, 84.0),
                "radius_px": 12.0,
                "confidence": 0.9,
                "candidates": ("cue",),
                "rejection_reason": None,
            },
        )


def test_classic_adapter_normalizes_existing_detector_output() -> None:
    adapter = ClassicCvAdapter(FakeClassicDetector())

    observations = adapter.detect(
        ({"frame_id": 7, "timestamp_ms": 240, "image": "synthetic-frame"},)
    )

    assert observations == (
        {
            "frame_id": 7,
            "timestamp_ms": 240,
            "center_px": [42.0, 84.0],
            "radius_px": 12.0,
            "confidence": 0.9,
            "identity_candidates": ["cue"],
            "supported": True,
            "abstention_reason": None,
            "detector_version": "fake-classic/1",
        },
    )


FELT_BGR = (72, 132, 58)


def _table_image(*, width: int = 600, height: int = 400) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = FELT_BGR
    return image


def _png_round_trip(image: np.ndarray) -> np.ndarray:
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    decoded = cv2.imdecode(png, cv2.IMREAD_COLOR)
    assert decoded is not None
    return decoded


def test_real_backend_detects_decoded_ball_pixels_and_dominant_colour_candidates() -> None:
    image = _table_image()
    cv2.circle(image, (150, 200), 22, (242, 242, 242), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (450, 200), 22, (20, 210, 235), thickness=-1, lineType=cv2.LINE_AA)

    detections = classic_cv.ClassicDetectorBackend().detect(_png_round_trip(image))

    assert len(detections) == 2
    cue, yellow = detections
    assert cue["center_px"] == pytest.approx((150.0, 200.0), abs=1.0)
    assert cue["radius_px"] == pytest.approx(22.0, abs=1.5)
    assert cue["candidates"] == ("cue",)
    assert cue["rejection_reason"] is None
    assert cue["circularity"] >= 0.82
    assert len(cue["mask_polygon_px"]) >= 8
    assert yellow["center_px"] == pytest.approx((450.0, 200.0), abs=1.0)
    assert yellow["candidates"] == ("1", "9")
    assert yellow["rejection_reason"] is None


def test_real_adapter_exposes_contour_mask_as_structured_prediction_evidence() -> None:
    image = _table_image()
    cv2.circle(image, (300, 200), 22, (242, 242, 242), thickness=-1, lineType=cv2.LINE_AA)
    adapter = ClassicCvAdapter(classic_cv.ClassicDetectorBackend())

    prediction = adapter.detect(
        ({"frame_id": 3, "timestamp_ms": 120, "image": _png_round_trip(image)},)
    )[0]

    assert prediction["mask"]["format"] == "polygon_px"
    assert len(prediction["mask"]["points"]) >= 8
    assert prediction["circularity"] >= 0.82
    assert prediction["dominant_color"] == "white"


def test_real_backend_confidence_is_local_to_each_contour() -> None:
    weak_only = _table_image()
    cv2.circle(weak_only, (150, 200), 22, (90, 155, 70), thickness=-1, lineType=cv2.LINE_AA)
    with_strong = weak_only.copy()
    cv2.circle(with_strong, (450, 200), 22, (242, 242, 242), thickness=-1, lineType=cv2.LINE_AA)
    backend = classic_cv.ClassicDetectorBackend()

    alone = backend.detect(_png_round_trip(weak_only))[0]["confidence"]
    alongside = next(
        detection["confidence"]
        for detection in backend.detect(_png_round_trip(with_strong))
        if detection["center_px"][0] < 300
    )

    assert alongside == pytest.approx(alone, abs=1e-12)


def test_real_backend_filters_contours_outside_the_ball_radius_range() -> None:
    image = _table_image()
    cv2.circle(image, (150, 200), 5, (242, 242, 242), thickness=-1)
    cv2.circle(image, (450, 200), 50, (242, 242, 242), thickness=-1)

    assert classic_cv.ClassicDetectorBackend().detect(_png_round_trip(image)) == ()


def test_real_backend_abstains_from_non_circular_occluded_blob() -> None:
    image = _table_image()
    cv2.circle(image, (300, 200), 22, (205, 80, 35), thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (300, 175), (330, 225), FELT_BGR, thickness=-1)

    detections = classic_cv.ClassicDetectorBackend().detect(_png_round_trip(image))

    assert len(detections) == 1
    assert detections[0]["candidates"] == ("2", "10")
    assert detections[0]["circularity"] < 0.82
    assert detections[0]["rejection_reason"] == "occluded"


def test_synthetic_baseline_runs_real_inference_and_records_failures_and_abstention() -> None:
    evidence = classic_cv.run_synthetic_classic_cv_baseline()

    assert evidence["status"] == "PASS_SYNTHETIC_REGRESSION_ONLY"
    assert evidence["candidate_id"] == "classic-cv-local-contrast-v1"
    assert evidence["input"] == {"synthetic_only": True, "real_video_count": 0}
    assert evidence["runtime"]["opencv"] == cv2.__version__
    assert evidence["implementation"]["synthetic_truth_adapter"] is False
    assert evidence["summary"] == {
        "case_count": 5,
        "supported_prediction_count": 5,
        "abstained_prediction_count": 1,
        "filtered_failure_count": 2,
    }
    cases = {case["case_id"]: case for case in evidence["cases"]}
    assert cases["nominal-colours"]["metrics"] == {
        "expected_ball_count": 3,
        "matched_ball_count": 3,
        "recall": 1.0,
        "maximum_center_error_px": pytest.approx(0.0, abs=1.0),
    }
    assert cases["blank-table"]["predictions"] == []
    assert cases["radius-filter"]["predictions"] == []
    assert cases["occluded-abstain"]["predictions"][0]["abstention_reason"] == "occluded"
    assert cases["local-confidence"]["metrics"]["weak_confidence_delta"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_classic_cv_baseline_cli_writes_structured_evidence(tmp_path: Path) -> None:
    output = tmp_path / "classic-cv-evidence.json"

    exit_code = main(["classic-cv-synthetic-baseline", "--output", str(output)])

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert evidence["status"] == "PASS_SYNTHETIC_REGRESSION_ONLY"
    assert evidence["summary"]["case_count"] == 5
