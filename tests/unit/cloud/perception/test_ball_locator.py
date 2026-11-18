"""Synthetic-only tests for the classic-CV ball locator service."""

import hashlib
from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

NOW = datetime(2026, 7, 28, 12, tzinfo=UTC)
DEVICE_ID = "overhead-test-device"
IMAGE_WIDTH = 2800
IMAGE_HEIGHT = 1500
TABLE_CORNERS = ((100.0, 100.0), (2640.0, 100.0), (2640.0, 1370.0), (100.0, 1370.0))


def _context():
    from billiards_ai.shared.models.calibration import (
        CalibrationResolution,
        CameraProfile,
        DistortionModel,
        DistortionParameters,
        TableProfile,
    )

    return {
        "camera_profile": CameraProfile(
            camera_profile_id="synthetic-overhead-v1",
            position_fingerprint="a" * 64,
            lens_fingerprint="b" * 64,
        ),
        "resolution": CalibrationResolution(width_px=IMAGE_WIDTH, height_px=IMAGE_HEIGHT),
        "table_profile": TableProfile(
            table_profile_id="synthetic-9ft-v1",
            length_mm=2540,
            width_mm=1270,
            configuration_fingerprint="c" * 64,
        ),
        "distortion": DistortionParameters(model=DistortionModel.NONE, verified=True),
    }


def _calibration_service(tmp_path, *, create: bool = True):
    from billiards_ai.cloud.calibration import CalibrationService, JsonCalibrationRepository
    from billiards_ai.shared.models import Point
    from billiards_ai.shared.models.calibration import CalibrationReference

    service = CalibrationService(JsonCalibrationRepository(tmp_path / "calibration" / "store.json"))
    if create:
        service.create_calibration(
            device_id=DEVICE_ID,
            **_context(),
            corners_px=TABLE_CORNERS,
            source_image_sha256="d" * 64,
            references=(
                CalibrationReference(pixel=(1370.0, 735.0), table=Point(x_mm=1270, y_mm=635)),
            ),
            validated_at=NOW,
        )
    return service


def _media_contract():
    from billiards_ai.shared.models.media import (
        FrameManifest,
        FrameRateStrategy,
        FrameRecord,
        MediaEquivalenceRule,
        MediaResolution,
        NormalizedMedia,
        TimeBase,
        ToolVersions,
    )

    frame_hash = "1" * 64
    sequence_hash = hashlib.sha256(f"{frame_hash}\n".encode()).hexdigest()
    tools = ToolVersions(ffmpeg="7.1", ffprobe="7.1", decoder="ffmpeg-png-7.1")
    common = {
        "input_sha256": "e" * 64,
        "processing_config_sha256": "f" * 64,
        "tool_versions": tools,
        "equivalence_rule": MediaEquivalenceRule.DECODED_FRAME_SHA256_V1,
        "decoded_frames_sha256": sequence_hash,
    }
    manifest = FrameManifest(
        **common,
        interval_ms=250,
        source_time_base=TimeBase(numerator=1, denominator=1000),
        first_source_pts=1000,
        frames=(
            FrameRecord(
                index=0,
                requested_timestamp_ms=0,
                source_pts=1000,
                image_path="frames/000000.png",
                content_sha256=frame_hash,
            ),
        ),
    )
    normalized = NormalizedMedia(
        **common,
        media_path="normalized.mp4",
        container="mp4",
        video_codec="h264",
        video_profile="High",
        pixel_format="yuv420p",
        resolution=MediaResolution(width_px=IMAGE_WIDTH, height_px=IMAGE_HEIGHT),
        source_orientation_degrees=0,
        orientation_degrees=0,
        time_base=TimeBase(numerator=1, denominator=1000),
        frame_rate_strategy=FrameRateStrategy.PRESERVE_VFR_SAMPLE_BY_PTS,
        audio_removed=True,
        content_sha256="0" * 64,
    )
    return normalized, manifest


def _blank_table(brightness_scale: float = 1.0) -> np.ndarray:
    felt = np.asarray((72, 132, 58), dtype=np.float32) * brightness_scale
    image = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:] = np.clip(felt, 0, 255).astype(np.uint8)
    return image


BALL_COLORS_BGR = (
    (242, 242, 242),
    (20, 210, 235),
    (205, 80, 35),
    (35, 35, 220),
    (155, 55, 145),
    (25, 130, 240),
    (35, 105, 25),
    (55, 45, 125),
    (20, 20, 20),
    (30, 190, 220),
)


def _draw_balls(image: np.ndarray, centers: tuple[tuple[int, int], ...], *, radius=22):
    for center, color in zip(centers, BALL_COLORS_BGR, strict=True):
        cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def _locate(tmp_path, image, *, service=None, expected_ball_ids=None):
    from billiards_ai.cloud.perception import BallLocatorService, ClassicCvBallDetector

    normalized, manifest = _media_contract()
    locator = BallLocatorService(
        calibration_service=service or _calibration_service(tmp_path),
        detector=ClassicCvBallDetector(),
    )
    arguments = {}
    if expected_ball_ids is not None:
        arguments["expected_ball_ids"] = expected_ball_ids
    return locator.locate_frame(
        normalized_media=normalized,
        manifest=manifest,
        frame_index=0,
        image_bgr=image,
        device_id=DEVICE_ID,
        **_context(),
        at=NOW,
        **arguments,
    )


@pytest.mark.parametrize("brightness_scale", [0.72, 1.0, 1.28])
def test_synthetic_standard_light_detection_recall_and_position_error_meet_gate(
    tmp_path, brightness_scale
):
    centers = (
        (350, 300),
        (700, 300),
        (1050, 300),
        (1400, 300),
        (1750, 300),
        (2100, 300),
        (450, 850),
        (900, 850),
        (1500, 850),
        (2250, 850),
    )
    image = _blank_table(brightness_scale)
    _draw_balls(image, centers)

    result = _locate(tmp_path, image, expected_ball_ids=())
    expected_mm = tuple((x - 100, y - 100) for x, y in centers)
    errors = []
    for expected in expected_mm:
        nearest = min(
            np.hypot(
                detection.position.x_mm - expected[0],
                detection.position.y_mm - expected[1],
            )
            for detection in result.detections
        )
        if nearest <= 20.0:
            errors.append(float(nearest))

    recall = len(errors) / len(expected_mm)
    assert recall >= 0.90
    assert max(errors) <= 20.0
    assert all(detection.position.is_on_table() for detection in result.detections)


def test_missing_or_inactive_calibration_fails_closed_even_for_empty_frame(tmp_path):
    from billiards_ai.cloud.calibration import CalibrationInvalidError, NotCalibratedError

    with pytest.raises(NotCalibratedError):
        _locate(
            tmp_path,
            _blank_table(),
            service=_calibration_service(tmp_path, create=False),
        )

    inactive = _calibration_service(tmp_path)
    inactive.invalidate(DEVICE_ID, reason="synthetic camera moved", at=NOW)
    with pytest.raises(CalibrationInvalidError):
        _locate(tmp_path, _blank_table(), service=inactive)


def test_locator_uses_actual_source_pts_not_sampling_target_as_frame_timestamp(tmp_path):
    from billiards_ai.cloud.perception import BallLocatorService, ClassicCvBallDetector
    from billiards_ai.shared.models.media import FrameRecord

    normalized, manifest = _media_contract()
    manifest = manifest.model_copy(
        update={
            "frames": (
                FrameRecord(
                    index=0,
                    requested_timestamp_ms=0,
                    source_pts=1040,
                    image_path="frames/000000.png",
                    content_sha256="1" * 64,
                ),
            )
        }
    )
    locator = BallLocatorService(
        calibration_service=_calibration_service(tmp_path), detector=ClassicCvBallDetector()
    )

    result = locator.locate_frame(
        normalized_media=normalized,
        manifest=manifest,
        frame_index=0,
        image_bgr=_blank_table(),
        device_id=DEVICE_ID,
        **_context(),
        at=NOW,
        expected_ball_ids=(),
    )

    assert result.timestamp_ms == 40


def test_locator_binds_the_exact_calibration_lineage_to_every_millimetre_frame(tmp_path):
    service = _calibration_service(tmp_path)
    expected = service.require_active(device_id=DEVICE_ID, **_context(), at=NOW)
    result = _locate(tmp_path, _blank_table(), service=service, expected_ball_ids=())

    assert hasattr(result, "calibration")
    assert result.calibration.calibration_id == expected.calibration_id
    assert result.calibration.revision_id == expected.revision_id
    assert result.calibration.revision_number == expected.revision_number
    assert (
        result.calibration.table_configuration_fingerprint
        == expected.table_profile.configuration_fingerprint
    )


def test_outside_table_and_occluded_shapes_are_rejected_without_guessed_positions(tmp_path):
    from billiards_ai.shared.models.detection import RejectionReason

    image = _blank_table()
    visible_center = (800, 600)
    outside_center = (2720, 600)
    occluded_center = (1800, 600)
    cv2.circle(image, visible_center, 22, BALL_COLORS_BGR[0], -1, lineType=cv2.LINE_AA)
    cv2.circle(image, outside_center, 22, BALL_COLORS_BGR[8], -1, lineType=cv2.LINE_AA)
    cv2.circle(image, occluded_center, 22, BALL_COLORS_BGR[2], -1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (1800, 575), (1830, 625), tuple(int(v) for v in image[0, 0]), -1)

    result = _locate(tmp_path, image, expected_ball_ids=(0, 2, 8))

    assert len(result.detections) == 1
    assert result.detections[0].position.x_mm == pytest.approx(700, abs=3)
    assert {item.reason for item in result.rejected} >= {
        RejectionReason.OUTSIDE_TABLE,
        RejectionReason.OCCLUDED,
    }
    assert all(item.position is None for item in result.rejected)
    assert 2 in result.missing_ball_ids
    assert 8 in result.missing_ball_ids


def test_media_contract_mismatch_and_wrong_image_shape_are_rejected(tmp_path):
    from billiards_ai.cloud.perception import BallLocatorService, ClassicCvBallDetector
    from billiards_ai.cloud.perception.errors import DetectionInputError

    normalized, manifest = _media_contract()
    locator = BallLocatorService(
        calibration_service=_calibration_service(tmp_path),
        detector=ClassicCvBallDetector(),
    )
    bad_media = normalized.model_copy(update={"processing_config_sha256": "9" * 64})
    arguments = {
        "manifest": manifest,
        "frame_index": 0,
        "image_bgr": _blank_table(),
        "device_id": DEVICE_ID,
        **_context(),
        "at": NOW,
    }

    with pytest.raises(DetectionInputError, match="media contract"):
        locator.locate_frame(normalized_media=bad_media, **arguments)

    with pytest.raises(DetectionInputError, match="resolution"):
        locator.locate_frame(
            normalized_media=normalized,
            **(arguments | {"image_bgr": np.zeros((100, 100, 3), dtype=np.uint8)}),
        )

    bad_resolution = normalized.model_copy(
        update={"resolution": normalized.resolution.model_copy(update={"width_px": 100})}
    )
    with pytest.raises(DetectionInputError, match="calibration resolution"):
        locator.locate_frame(
            normalized_media=bad_resolution,
            **(arguments | {"image_bgr": np.zeros((IMAGE_HEIGHT, 100, 3), dtype=np.uint8)}),
        )


def test_full_decode_and_sampled_frame_hashes_may_differ(tmp_path):
    from billiards_ai.cloud.perception import BallLocatorService, ClassicCvBallDetector

    normalized, manifest = _media_contract()
    normalized = normalized.model_copy(update={"decoded_frames_sha256": "8" * 64})
    locator = BallLocatorService(
        calibration_service=_calibration_service(tmp_path),
        detector=ClassicCvBallDetector(),
    )

    result = locator.locate_frame(
        normalized_media=normalized,
        manifest=manifest,
        frame_index=0,
        image_bgr=_blank_table(),
        device_id=DEVICE_ID,
        **_context(),
        at=NOW,
    )

    assert result.detections == ()
