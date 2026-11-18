from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


class FakeFfprobe:
    def run(self, _path: Path) -> dict[str, object]:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "profile": "High",
                    "pix_fmt": "yuv420p",
                    "width": 192,
                    "height": 108,
                    "r_frame_rate": "30/1",
                    "avg_frame_rate": "30000/1001",
                    "time_base": "1/90000",
                    "nb_frames": "8",
                    "duration": "0.266933",
                    "tags": {"rotate": "90"},
                },
                {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "0.266933"},
            "frames": [
                {"best_effort_timestamp_time": "0.000000"},
                {"best_effort_timestamp_time": "0.033367"},
                {"best_effort_timestamp_time": "0.066733"},
            ],
        }


def _tiny_video(path: Path) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for index in range(8):
        frame = np.full((48, 64, 3), index * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_probe_records_codec_timestamps_and_orientation(tmp_path: Path) -> None:
    from billiards_ai.lab.video_probe import probe_video

    probe = probe_video(tmp_path / "tiny.mp4", FakeFfprobe())

    assert probe.width == 192
    assert probe.height == 108
    assert probe.codec == "h264"
    assert probe.average_fps == pytest.approx(29.97002997)
    assert probe.audio_stream_count == 1
    assert probe.timestamps_monotonic is True
    assert probe.rotation_degrees == 90


def test_probe_flags_non_monotonic_timestamps(tmp_path: Path) -> None:
    from billiards_ai.lab.video_probe import probe_video

    ffprobe = FakeFfprobe()
    payload = ffprobe.run(tmp_path / "tiny.mp4")
    payload["frames"] = [
        {"best_effort_timestamp_time": "0.1"},
        {"best_effort_timestamp_time": "0.0"},
    ]
    ffprobe.run = lambda _path: payload  # type: ignore[method-assign]

    assert probe_video(tmp_path / "tiny.mp4", ffprobe).timestamps_monotonic is False


def test_probe_ignores_interleaved_audio_timestamps(tmp_path: Path) -> None:
    from billiards_ai.lab.video_probe import probe_video

    ffprobe = FakeFfprobe()
    payload = ffprobe.run(tmp_path / "tiny.mp4")
    payload["frames"] = [
        {"media_type": "video", "best_effort_timestamp_time": "0.0"},
        {"media_type": "audio", "best_effort_timestamp_time": "-0.1"},
        {"media_type": "video", "best_effort_timestamp_time": "0.033"},
    ]
    ffprobe.run = lambda _path: payload  # type: ignore[method-assign]

    assert probe_video(tmp_path / "tiny.mp4", ffprobe).timestamps_monotonic is True


def test_sampler_uses_decoded_frame_index_not_seek_approximation(tmp_path: Path) -> None:
    from billiards_ai.lab.video_probe import sample_frames

    video = _tiny_video(tmp_path / "tiny.mp4")

    frames = sample_frames(video, (0, 3, 7))

    assert [frame.frame_index for frame in frames] == [0, 3, 7]
    assert [round(float(frame.image.mean()) / 25) for frame in frames] == [0, 3, 7]
    assert all(frame.quality.luma_mean >= 0 for frame in frames)


def test_sampler_rejects_missing_frame(tmp_path: Path) -> None:
    from billiards_ai.lab.video_probe import VideoProbeError, sample_frames

    video = _tiny_video(tmp_path / "tiny.mp4")

    with pytest.raises(VideoProbeError, match="not decoded"):
        sample_frames(video, (99,))
