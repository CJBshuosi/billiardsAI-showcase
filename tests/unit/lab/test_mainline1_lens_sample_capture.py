from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_lens_sample_capture import (
    CheckerboardSpec,
    LensSampleCaptureError,
    collect_usb_checkerboard_samples,
    detect_checkerboard_sample,
    select_checkerboard_roi,
)


def test_detect_checkerboard_sample_builds_metric_object_points() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    corners = np.asarray(
        [[[100.0 + column * 20.0, 200.0 + row * 20.0]] for row in range(6) for column in range(9)],
        dtype=np.float32,
    )

    sample = detect_checkerboard_sample(
        frame,
        CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
        detector=lambda _gray, _pattern, **_kwargs: (True, corners),
    )

    assert sample is not None
    assert len(sample.image_points) == 54
    assert sample.image_points[0] == (100.0, 200.0)
    assert sample.object_points[0] == (0.0, 0.0, 0.0)
    assert sample.object_points[8] == (200.0, 0.0, 0.0)
    assert sample.object_points[-1] == (200.0, 125.0, 0.0)


def test_detect_checkerboard_sample_returns_none_when_pattern_is_absent() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    sample = detect_checkerboard_sample(
        frame,
        CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
        detector=lambda _gray, _pattern, **_kwargs: (False, None),
    )

    assert sample is None


def test_roi_detection_keeps_full_frame_coordinates_and_metric_points():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.array(
        [[[10 + c * 20, 15 + r * 20]] for r in range(8) for c in range(11)], dtype=np.float32
    )

    def detector(gray, pattern, **kwargs):
        assert gray.shape == (190, 260)
        assert pattern == (11, 8)
        return True, corners

    sample = detect_checkerboard_sample(
        frame, CheckerboardSpec(11, 8, 25), detector=detector, roi=(100, 80, 260, 190)
    )
    assert sample.image_points[0] == (110.0, 95.0)
    assert sample.image_points[-1] == (310.0, 235.0)
    assert sample.object_points[-1] == (250.0, 175.0, 0.0)


@pytest.mark.parametrize(
    "roi", [(-1, 0, 20, 20), (0, 0, 0, 20), (630, 0, 20, 20), (0, 470, 20, 20), (0.5, 0, 20, 20)]
)
def test_roi_outside_original_frame_is_rejected(roi):
    with pytest.raises(ValueError, match="ROI"):
        detect_checkerboard_sample(
            np.zeros((480, 640, 3), dtype=np.uint8), CheckerboardSpec(11, 8, 25), roi=roi
        )


def test_partial_or_outside_roi_corners_are_not_accepted():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.array(
        [[[10 + c * 20, 15 + r * 20]] for r in range(8) for c in range(11)], dtype=np.float32
    )
    corners[-1] = [[261, 10]]
    assert (
        detect_checkerboard_sample(
            frame,
            CheckerboardSpec(11, 8, 25),
            roi=(100, 80, 260, 190),
            detector=lambda *_a, **_k: (True, corners),
        )
        is None
    )


def test_r_key_selects_roi_but_does_not_save_the_frozen_selection_frame(tmp_path):
    capture = _Capture()
    count = [0]

    def read():
        count[0] += 1
        return True, np.full((1080, 1920, 3), count[0], dtype=np.uint8)

    capture.read = read
    keys = iter([ord("r")] + [ord(" ")] * 8)
    corners = np.array(
        [[[10 + c * 20, 15 + r * 20]] for r in range(6) for c in range(9)], dtype=np.float32
    )
    clock = iter([0, 16_666_667])
    seen = []

    def detector(gray, pattern, **kwargs):
        seen.append(gray.shape)
        return (False, None) if gray.shape == (1080, 1920) else (True, corners)

    evidence = {}
    samples = collect_usb_checkerboard_samples(
        runtime_index=0,
        spec=CheckerboardSpec(9, 6, 25),
        target_samples=8,
        capture_factory=lambda _: capture,
        detector=detector,
        key_reader=lambda _: next(keys),
        presenter=lambda *_: None,
        window_closer=lambda: None,
        monotonic_ns=lambda: next(clock),
        cadence_sample_frames=2,
        roi_selector=lambda frame: (100, 80, 260, 190),
        evidence_directory=tmp_path / "frames",
        capture_evidence=evidence,
    )
    assert len(samples) == 8
    assert samples[0].image_points[0] == (110.0, 95.0)
    assert seen == [(1080, 1920)] + [(190, 260)] * 8
    assert cv2.imread(str(tmp_path / "frames/sample-001.png")).shape == (1080, 1920, 3)
    assert np.all(cv2.imread(str(tmp_path / "frames/sample-001.png")) == 4)
    assert np.all(cv2.imread(str(tmp_path / "frames/roi-001.png")) == 3)
    assert (tmp_path / "frames/sample-001-corners.png").is_file()
    assert evidence["frame_evidence"][-1]["roi"] == [100, 80, 260, 190]
    sidecar = json.loads((tmp_path / "frames/sample-001.json").read_text())
    assert sidecar["coordinate_space"] == "original_full_frame_pixels"
    assert sidecar["image_points"][0] == [110.0, 95.0]
    for name, digest in sidecar["files"].items():
        assert hashlib.sha256((tmp_path / "frames" / name).read_bytes()).hexdigest() == digest


def test_cancelled_roi_selection_does_not_create_samples(tmp_path):
    capture = _Capture()
    keys = iter([ord("r"), ord("q")])
    clock = iter([0, 16_666_667])
    with pytest.raises(LensSampleCaptureError, match="cancelled"):
        collect_usb_checkerboard_samples(
            runtime_index=0,
            spec=CheckerboardSpec(9, 6, 25),
            target_samples=8,
            capture_factory=lambda _: capture,
            detector=lambda *_a, **_k: (False, None),
            key_reader=lambda _: next(keys),
            presenter=lambda *_: None,
            window_closer=lambda: None,
            monotonic_ns=lambda: next(clock),
            cadence_sample_frames=2,
            roi_selector=lambda _: (0, 0, 0, 0),
            evidence_directory=tmp_path / "frames",
        )
    assert capture.released
    assert (tmp_path / "frames/initial.png").is_file()
    assert not list((tmp_path / "frames").glob("sample-*"))


def test_real_opencv_roi_matches_independent_crop_plus_offset():
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[80:310, 160:450] = 255
    for row in range(9):
        for column in range(12):
            if (row + column) % 2 == 0:
                x, y = 185 + column * 20, 105 + row * 20
                frame[y : y + 20, x : x + 20] = 0
    spec = CheckerboardSpec(11, 8, 25)
    roi_sample = detect_checkerboard_sample(frame, spec, roi=(160, 80, 290, 230))
    crop_sample = detect_checkerboard_sample(frame[80:310, 160:450], spec)
    assert roi_sample is not None and crop_sample is not None
    assert len(roi_sample.image_points) == 88
    np.testing.assert_allclose(
        roi_sample.image_points, np.asarray(crop_sample.image_points) + [160, 80], atol=0.0001
    )


def test_existing_evidence_directory_is_preserved(tmp_path):
    original = tmp_path / "keep.txt"
    original.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        _collect_fixture(_Capture(), evidence_directory=tmp_path)
    assert original.read_text() == "prior evidence"


@pytest.mark.parametrize("fail", [False, True])
def test_roi_window_passes_original_pixels_and_closes(monkeypatch, fail):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    closed = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: closed.append(name))

    def select(name, pixels, **kwargs):
        assert pixels is frame
        if fail:
            raise RuntimeError("selection failure")
        return (100, 80, 260, 190)

    monkeypatch.setattr(cv2, "selectROI", select)
    if fail:
        with pytest.raises(RuntimeError, match="selection failure"):
            select_checkerboard_roi(frame)
    else:
        assert select_checkerboard_roi(frame) == (100, 80, 260, 190)
    assert len(closed) == 1


class _Capture:
    def __init__(self, *, width: int = 1920, height: int = 1080, fps: float = 60.0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def set(self, property_id: int, value: float) -> bool:
        return True

    def get(self, property_id: int) -> float:
        import cv2

        return {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
            cv2.CAP_PROP_FPS: self.fps,
        }.get(property_id, 0.0)

    def read(self):
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _collect_fixture(capture, **options):
    corners = np.array(
        [[[100 + c * 20, 200 + r * 20]] for r in range(6) for c in range(9)], dtype=np.float32
    )
    clock = iter(i * 16_666_667 for i in range(200))
    return collect_usb_checkerboard_samples(
        runtime_index=0,
        spec=CheckerboardSpec(9, 6, 25),
        target_samples=8,
        capture_factory=lambda _: capture,
        detector=lambda *_a, **_k: (True, corners),
        key_reader=lambda _: ord(" "),
        presenter=lambda *_: None,
        window_closer=lambda: None,
        monotonic_ns=lambda: next(clock),
        cadence_sample_frames=2,
        **options,
    )


def test_mjpg_is_set_after_fps_so_directshow_does_not_reset_it():
    capture = _Capture()
    format_value = [0]

    def setter(prop, value):
        if prop == cv2.CAP_PROP_FPS:
            format_value[0] = cv2.VideoWriter_fourcc(*"YUY2")
        elif prop == cv2.CAP_PROP_FOURCC:
            format_value[0] = value
        return True

    capture.set = setter
    _collect_fixture(capture)
    assert format_value[0] == cv2.VideoWriter_fourcc(*"MJPG")


def test_rer_capture_rejects_wrong_codec_and_releases():
    capture = _Capture()
    with pytest.raises(LensSampleCaptureError, match="MJPG"):
        _collect_fixture(capture, require_mjpg=True)
    assert capture.released


@pytest.mark.parametrize("change_after_reads", [0, 4])
def test_manual_controls_must_match_before_and_during_sampling(change_after_reads):
    capture = _Capture()
    original_get, original_read = capture.get, capture.read
    reads = [0]

    def get(prop):
        if prop == cv2.CAP_PROP_EXPOSURE:
            return -7 if reads[0] >= change_after_reads else -8
        return original_get(prop)

    def read():
        reads[0] += 1
        return original_read()

    capture.get, capture.read = get, read
    with pytest.raises(LensSampleCaptureError, match="control"):
        _collect_fixture(capture, expected_controls={cv2.CAP_PROP_EXPOSURE: -8})
    assert capture.released


def test_rer_success_returns_capture_readback_without_writing_controls():
    capture = _Capture()
    original_get = capture.get
    writes = []
    capture.set = lambda prop, value: writes.append(prop) or True
    capture.get = lambda prop: {
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_EXPOSURE: -8,
    }.get(prop, original_get(prop))
    evidence = {}
    assert (
        len(
            _collect_fixture(
                capture,
                require_mjpg=True,
                expected_controls={cv2.CAP_PROP_EXPOSURE: -8},
                capture_evidence=evidence,
            )
        )
        == 8
    )
    assert evidence["measured_fps"] == pytest.approx(60, abs=0.01)
    assert evidence["controls_after_capture"][str(cv2.CAP_PROP_EXPOSURE)] == -8
    assert cv2.CAP_PROP_EXPOSURE not in writes


def test_default_capture_clock_is_high_resolution_performance_counter():
    import inspect
    import time

    assert (
        inspect.signature(collect_usb_checkerboard_samples).parameters["monotonic_ns"].default
        is time.perf_counter_ns
    )


def test_camera_warmup_frames_are_not_included_in_cadence_measurement():
    capture = _Capture()
    original_read = capture.read
    reads = [0]

    def read():
        reads[0] += 1
        return original_read()

    capture.read = read
    evidence = {}
    _collect_fixture(capture, warmup_frames=30, capture_evidence=evidence)
    assert reads[0] == 40  # 30 warmup + 2 cadence + 8 samples
    assert evidence["warmup_frames"] == 30
    assert evidence["measured_fps"] == pytest.approx(60, abs=0.01)


def test_collect_usb_checkerboard_samples_saves_only_on_space() -> None:
    capture = _Capture()
    corners = np.asarray(
        [[[100.0 + column, 200.0 + row]] for row in range(6) for column in range(9)],
        dtype=np.float32,
    )
    keys = iter((-1, *(ord(" ") for _ in range(8))))
    clock = iter(index * 16_666_667 for index in range(130))

    samples = collect_usb_checkerboard_samples(
        runtime_index=0,
        spec=CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
        target_samples=8,
        capture_factory=lambda _index: capture,
        detector=lambda _gray, _pattern, **_kwargs: (True, corners),
        key_reader=lambda _delay: next(keys),
        presenter=lambda _name, _frame: None,
        window_closer=lambda: None,
        monotonic_ns=lambda: next(clock),
        cadence_sample_frames=120,
    )

    assert len(samples) == 8
    assert capture.released is True


def test_collect_usb_checkerboard_samples_rejects_driver_reported_sixty_when_slow() -> None:
    capture = _Capture()
    clock = iter(index * 33_333_333 for index in range(120))

    try:
        collect_usb_checkerboard_samples(
            runtime_index=0,
            spec=CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
            target_samples=8,
            capture_factory=lambda _index: capture,
            presenter=lambda _name, _frame: None,
            window_closer=lambda: None,
            monotonic_ns=lambda: next(clock),
            cadence_sample_frames=120,
        )
    except LensSampleCaptureError as exc:
        assert "delivery cadence" in str(exc)
    else:
        raise AssertionError("measured 30 fps delivery must fail closed")
    assert capture.released is True


def test_collect_usb_checkerboard_samples_rejects_wrong_frame_shape() -> None:
    capture = _Capture()
    capture.read = lambda: (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    clock = iter((0, 16_666_667))

    try:
        collect_usb_checkerboard_samples(
            runtime_index=0,
            spec=CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
            target_samples=8,
            capture_factory=lambda _index: capture,
            presenter=lambda _name, _frame: None,
            window_closer=lambda: None,
            cadence_sample_frames=2,
            monotonic_ns=lambda: next(clock),
        )
    except LensSampleCaptureError as exc:
        assert "frame geometry" in str(exc)
    else:
        raise AssertionError("wrong delivered frame geometry must fail closed")


def test_collect_usb_checkerboard_samples_rejects_silent_mode_fallback() -> None:
    capture = _Capture(width=1280, height=720)

    try:
        collect_usb_checkerboard_samples(
            runtime_index=0,
            spec=CheckerboardSpec(columns=9, rows=6, square_size_mm=25.0),
            target_samples=8,
            capture_factory=lambda _index: capture,
            presenter=lambda _name, _frame: None,
            window_closer=lambda: None,
        )
    except LensSampleCaptureError as exc:
        assert "1920x1080" in str(exc)
    else:
        raise AssertionError("capture mode fallback must fail closed")
    assert capture.released is True
