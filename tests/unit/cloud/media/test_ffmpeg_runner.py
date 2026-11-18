"""Typed failure behavior at the FFmpeg subprocess boundary."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest


def test_missing_ffmpeg_executable_is_explicit(monkeypatch):
    from billiards_ai.cloud.media import FFmpegUnavailableError, SubprocessFFmpegRunner
    from billiards_ai.cloud.media import runner as runner_module

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(runner_module.subprocess, "run", missing)

    with pytest.raises(FFmpegUnavailableError, match="ffmpeg"):
        SubprocessFFmpegRunner().tool_versions()


def test_unparseable_tool_version_is_explicit(monkeypatch):
    from billiards_ai.cloud.media import FFmpegVersionError, SubprocessFFmpegRunner
    from billiards_ai.cloud.media import runner as runner_module

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args=args[0], returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(FFmpegVersionError, match="version"):
        SubprocessFFmpegRunner().tool_versions()


def test_tool_versions_records_reported_libavcodec_version(monkeypatch):
    from billiards_ai.cloud.media import SubprocessFFmpegRunner
    from billiards_ai.cloud.media import runner as runner_module

    output = "ffmpeg version 7.1.1\nlibavcodec    61.19.100 / 61.19.100\n"
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=output if args[0][0] == "ffmpeg" else output.replace("ffmpeg", "ffprobe", 1),
            stderr="",
        ),
    )

    assert SubprocessFFmpegRunner().tool_versions().decoder == "libavcodec-61.19.100"


def test_probe_skips_attached_picture_and_reads_pts_from_the_real_video_stream(monkeypatch):
    from billiards_ai.cloud.media import SubprocessFFmpegRunner
    from billiards_ai.cloud.media import runner as runner_module

    calls: list[list[str]] = []
    metadata = """{"format":{"format_name":"mp4"},"streams":[
      {"codec_type":"video","codec_name":"mjpeg","disposition":{"attached_pic":1},
       "time_base":"1/90000","pix_fmt":"yuvj420p","width":600,"height":600},
      {"codec_type":"video","codec_name":"h264","profile":"High","time_base":"1/1000",
       "pix_fmt":"yuv420p","width":1920,"height":1080,"disposition":{"attached_pic":0}}
    ]}"""
    frames = '{"frames":[{"media_type":"video","best_effort_timestamp":"100"}]}'

    def run(args, **_kwargs):
        calls.append(args)
        return CompletedProcess(
            args=args,
            returncode=0,
            stdout=metadata if "-show_streams" in args else frames,
            stderr="",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", run)

    probe = SubprocessFFmpegRunner().probe(Path("input.mp4"))

    assert probe.video_stream_index == 1
    assert probe.frame_pts == (100,)
    assert calls[1][calls[1].index("-select_streams") + 1] == "v:1"


def test_subprocess_runner_applies_injected_timeout(monkeypatch):
    from billiards_ai.cloud.media import SubprocessFFmpegRunner, ToolExecutionError
    from billiards_ai.cloud.media import runner as runner_module

    def expired(*_args, **_kwargs):
        raise runner_module.subprocess.TimeoutExpired("ffmpeg", 3)

    monkeypatch.setattr(runner_module.subprocess, "run", expired)

    with pytest.raises(ToolExecutionError, match="timed out"):
        SubprocessFFmpegRunner(timeout_seconds=3).tool_versions()


def test_failed_normalization_raises_typed_tool_error(monkeypatch, tmp_path):
    from billiards_ai.cloud.media import (
        MediaProcessingConfig,
        SubprocessFFmpegRunner,
        ToolExecutionError,
    )
    from billiards_ai.cloud.media import runner as runner_module

    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="synthetic encoder failure"
        ),
    )

    with pytest.raises(ToolExecutionError, match="synthetic encoder failure"):
        SubprocessFFmpegRunner().normalize(
            Path("input.mp4"),
            tmp_path / "normalized.mp4",
            MediaProcessingConfig(interval_ms=100),
        )


@pytest.mark.parametrize("field,value", [("container", "mkv"), ("video_codec", "hevc")])
def test_unsupported_normalization_contract_is_rejected(field, value):
    from billiards_ai.cloud.media import MediaProcessingConfig

    with pytest.raises(ValueError):
        MediaProcessingConfig(interval_ms=100, **{field: value})
