from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.shot_local.active_tracks import (
    ActiveBallTrack,
    ActiveTrackSample,
    CushionContactDecision,
    DelayedCollisionDecision,
)
from billiards_ai.lab.shot_local.models import BallGroup, ShotLocalResult, ShotWindow
from billiards_ai.lab.shot_local.observations import DecodedTrackingFrame
from billiards_ai.lab.shot_local.product import (
    CanonicalShotAnalysis,
    canonical_event_key,
    canonical_world_event_rows,
)
from billiards_ai.lab.shot_local.product_overlay import (
    calibration_overlay_geometry,
    overlay_annotations_for_frame,
    render_canonical_overlay,
)
from billiards_ai.shared.calibration.homography import calibrate


def _calibration():
    return calibrate(
        "synthetic-camera",
        ((0, 120), (160, 120), (160, 0), (0, 0)),
        table_length_mm=160,
        table_width_mm=120,
    )


def _analysis() -> CanonicalShotAnalysis:
    track = ActiveBallTrack(
        shot_id="shot-0001",
        track_id="shot-0001-anchor-10",
        group=BallGroup.STRIPE,
        samples=(
            ActiveTrackSample(
                frame_id=71,
                timestamp_ms=2367,
                x_mm=100,
                y_mm=200,
                vx_mm_per_frame=4,
                vy_mm_per_frame=2,
                state="observed",
                confidence=0.95,
                source_detection_id="detection-71",
            ),
            ActiveTrackSample(
                frame_id=72,
                timestamp_ms=2400,
                x_mm=104,
                y_mm=202,
                vx_mm_per_frame=4,
                vy_mm_per_frame=2,
                state="reacquired",
                confidence=0.91,
                source_detection_id="detection-72",
            ),
        ),
    )
    evidence = ShotLocalResult(
        shot_id="shot-0001",
        window=ShotWindow(
            shot_id="shot-0001",
            start_frame_id=10,
            end_frame_id=80,
            before_frame_id=9,
            after_frame_id=81,
            start_ms=333,
            end_ms=2667,
            complete=True,
        ),
        anchors=(),
        cue_samples=(),
        fragments=(),
        events=(),
    )
    return CanonicalShotAnalysis(
        evidence=evidence,
        active_tracks=(track,),
        collisions=(
            DelayedCollisionDecision(
                actor_id="shot-0001-anchor-09",
                target_id="shot-0001-anchor-10",
                contact_frame_id=70,
                decision_frame_id=72,
                status="observed",
                confidence=0.98,
                evidence_frame_ids=(69, 70, 72),
                detection_layer="L1",
                contact_kind="new_contact",
                minimum_distance_mm=69.4,
            ),
        ),
        cushions=(
            CushionContactDecision(
                actor_id="shot-0001-anchor-10",
                rail="left",
                contact_frame_id=70,
                decision_frame_id=72,
                detection_layer="L3",
                confidence=0.82,
                minimum_distance_mm=47.3,
                contact_kind="trajectory_fitted_rebound",
                evidence_frame_ids=(68, 70, 72),
            ),
        ),
        retained_events=(),
        fragment_lineage={},
    )


def test_overlay_annotations_show_stable_state_trail_and_canonical_events() -> None:
    annotations = overlay_annotations_for_frame((_analysis(),), frame_id=72)

    assert annotations.track_labels == (("shot-0001-anchor-10", "reacquired"),)
    assert annotations.event_banners == (
        "COLLISION L1 shot-0001-anchor-09 -> shot-0001-anchor-10",
        "CUSHION L3 shot-0001-anchor-10 -> left",
    )
    assert annotations.trails[0].track_id == "shot-0001-anchor-10"
    assert len(annotations.trails[0].points_mm) == 2


def test_every_admitted_overlay_event_has_an_identical_world_event_row() -> None:
    analysis = _analysis()
    rows = canonical_world_event_rows(analysis)
    annotations = overlay_annotations_for_frame((analysis,), frame_id=72)

    assert {marker.event_key for marker in annotations.current_events} <= {
        canonical_event_key(row) for row in rows
    }


def test_overlay_has_no_fragment_identity_surface() -> None:
    annotations = overlay_annotations_for_frame((_analysis(),), frame_id=72)

    assert "fragment" not in repr(annotations)


def test_canonical_calibration_overlay_exposes_boundary_band_and_pockets() -> None:
    geometry = calibration_overlay_geometry(_calibration(), cushion_distance_mm=10)

    assert geometry["boundary"] == ((0, 120), (160, 120), (160, 0), (0, 0))
    assert geometry["cushion_band"] == ((10, 110), (150, 110), (150, 10), (10, 10))
    assert geometry["pockets"] == (
        (0, 120),
        (80, 120),
        (160, 120),
        (0, 0),
        (80, 0),
        (160, 0),
    )


def test_canonical_overlay_encodes_real_h264_with_all_frames_and_dimensions(tmp_path: Path) -> None:
    # An unavailable local encoder is an uncovered positive, not a fallback PASS.
    try:
        probe = cv2.VideoWriter(
            str(tmp_path / "encoder-probe.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 60, (320, 240)
        )
    except cv2.error:
        pytest.skip("local H.264 encoder unavailable; actual H.264 output not covered")
    try:
        if not probe.isOpened():
            pytest.skip("local H.264 encoder unavailable; actual H.264 output not covered")
    finally:
        probe.release()
    frames = tuple(
        DecodedTrackingFrame(
            frame_id=index,
            timestamp_ms=round(index * 1000 / 60),
            image=np.full((240, 320, 3), (index * 20, 80, 100), dtype=np.uint8),
        )
        for index in range(10)
    )
    output = tmp_path / "canonical.mp4"
    render_canonical_overlay(
        decoded_frames=frames,
        analyses=(),
        calibration=_calibration(),
        cushion_distance_mm=10,
        fps=60,
        output=output,
    )
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 255) for index in range(4))
        assert codec.lower() in {"avc1", "h264"}, f"incompatible replay codec: {codec}"
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 320
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 240
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(60)
        decoded_count = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (240, 320, 3)
            decoded_count += 1
        assert decoded_count == len(frames)
    finally:
        capture.release()


def test_unavailable_overlay_encoder_fails_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableWriter:
        released = False

        def isOpened(self):
            return False

        def write(self, image):
            pytest.fail("an unavailable encoder must never receive frames")

        def release(self):
            self.released = True

    writer = UnavailableWriter()
    attempts = []

    def unavailable(*args):
        attempts.append(args)
        return writer

    monkeypatch.setattr(cv2, "VideoWriter", unavailable)
    output = tmp_path / "unavailable.mp4"
    with pytest.raises(ValueError) as failure:
        render_canonical_overlay(
            decoded_frames=(DecodedTrackingFrame(0, 0, np.zeros((120, 160, 3), np.uint8)),),
            analyses=(),
            calibration=_calibration(),
            cushion_distance_mm=10,
            fps=60,
            output=output,
        )
    assert len(attempts) == 1
    assert writer.released
    assert "H.264" in str(failure.value) and "unavailable" in str(failure.value)
    assert not output.exists()
