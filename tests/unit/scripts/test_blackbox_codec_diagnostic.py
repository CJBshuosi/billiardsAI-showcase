import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame

ENTRY = Path(__file__).resolve().parents[3] / "scripts/development/blackbox_codec_diagnostic.py"


def module():
    assert ENTRY.is_file(), "codec comparison probe has not been implemented"
    spec = importlib.util.spec_from_file_location("codec_probe_test", ENTRY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def interval(*, fail_at=None):
    def frames(**kwargs):
        for i in range(40):
            if i == fail_at:
                raise ValueError("source failure")
            ok, jpeg = cv2.imencode(".jpg", np.full((16, 16, 3), i, dtype=np.uint8))
            assert ok
            yield CaptureFrame(
                source_epoch_id="test-codec-sample",
                capture_frame_id=i,
                captured_at_ns=i * 16666667,
                width=16,
                height=16,
                mjpeg=jpeg.tobytes(),
            )

    return SimpleNamespace(width=16, height=16, frame_count=40, frames=frames)


def factory(*, forced_open=True):
    writers = []

    def build(*args):
        opened = len(args) == 4 or forced_open

        class Writer:
            def __init__(self):
                self.frames = []
                self.releases = 0
                self.args = args

            def isOpened(self):
                return opened

            def getBackendName(self):
                return "FFMPEG"

            def write(self, image):
                self.frames.append(image.copy())

            def release(self):
                self.releases += 1

        writer = Writer()
        writers.append(writer)
        return writer

    return build, writers


def test_both_encoders_receive_identical_frames_and_explicit_api_is_bound(tmp_path):
    m = module()
    build, writers = factory()
    rows = []
    p = m.helper.Profiler(rows.append)
    result = m.encode_pair(interval(), tmp_path, 60, p, writer_factory=build)
    assert len(writers[0].args) == 4
    assert writers[1].args[1] == cv2.CAP_FFMPEG
    assert [len(w.frames) for w in writers] == [40, 40]
    assert all(
        np.array_equal(a, b) for a, b in zip(writers[0].frames, writers[1].frames, strict=True)
    )
    assert [w.releases for w in writers] == [1, 1]
    assert all(row["frames"] == 40 for row in result)


def test_unavailable_forced_backend_is_reported_without_auto_fallback(tmp_path):
    m = module()
    build, writers = factory(forced_open=False)
    p = m.helper.Profiler(lambda row: None)
    result = m.encode_pair(interval(), tmp_path, 60, p, writer_factory=build)
    assert len(writers) == 2
    assert result[1]["opened"] is False
    assert writers[1].frames == [] and writers[1].releases == 1
    assert len(writers[0].frames) == 40


def test_both_writers_release_on_original_frame_error(tmp_path):
    m = module()
    build, writers = factory()
    p = m.helper.Profiler(lambda row: None)
    with pytest.raises(ValueError, match="source failure"):
        m.encode_pair(interval(fail_at=3), tmp_path, 60, p, writer_factory=build)
    assert [w.releases for w in writers] == [1, 1]
