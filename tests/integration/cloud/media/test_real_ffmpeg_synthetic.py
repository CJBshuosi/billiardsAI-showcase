"""Small real-toolchain check using generated pixels and tones only."""

import shutil
import subprocess

import pytest


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg toolchain is not installed",
)
def test_real_ffmpeg_normalizes_synthetic_vfr_and_removes_audio(tmp_path):
    from billiards_ai.cloud.media import (
        MediaProcessingConfig,
        MediaProcessingLimits,
        MediaProcessingService,
        SubprocessFFmpegRunner,
    )

    source = tmp_path / "synthetic-vfr-with-audio.mp4"
    generated = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=5:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=8000:duration=2",
            "-vf",
            "setpts=N*N/(10*TB)",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

    service = MediaProcessingService(
        SubprocessFFmpegRunner(timeout_seconds=30),
        limits=MediaProcessingLimits(
            max_input_bytes=100_000_000,
            max_decoded_frames=10_000,
            max_sampled_frames=1_000,
        ),
    )
    assert service.runner.probe(source).audio_present is True
    config = MediaProcessingConfig(interval_ms=250)
    first = service.process(input_path=source, output_dir=tmp_path / "first", config=config)
    second = service.process(input_path=source, output_dir=tmp_path / "second", config=config)

    assert first.frame_manifest == second.frame_manifest
    assert first.frame_manifest.frames[0].requested_timestamp_ms == 0
    assert [frame.source_pts for frame in first.frame_manifest.frames] != list(
        range(len(first.frame_manifest.frames))
    )
    normalized_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(tmp_path / "first" / first.normalized_media.media_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert normalized_probe.returncode == 0
    assert normalized_probe.stdout.strip() == ""
    normalized_path = tmp_path / "first" / first.normalized_media.media_path
    assert service.runner.probe(normalized_path).audio_present is False
