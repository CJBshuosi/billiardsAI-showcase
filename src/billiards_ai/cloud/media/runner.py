"""Injectable FFmpeg/ffprobe boundary and its subprocess implementation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from billiards_ai.shared.models.calibration import StrictFrozenModel
from billiards_ai.shared.models.media import MediaResolution, TimeBase, ToolVersions

from .errors import (
    CorruptMediaError,
    FFmpegUnavailableError,
    FFmpegVersionError,
    ToolExecutionError,
    UnsupportedMediaError,
)


class MediaProcessingConfig(StrictFrozenModel):
    interval_ms: int = Field(gt=0)
    width_px: int = Field(default=1280, gt=0)
    height_px: int = Field(default=720, gt=0)
    container: Literal["mp4"] = "mp4"
    video_codec: Literal["h264"] = "h264"
    encoder: Literal["libx264"] = "libx264"
    video_profile: Literal["High"] = "High"
    pixel_format: Literal["yuv420p"] = "yuv420p"


class MediaProbe(StrictFrozenModel):
    container: str
    video_codec: str
    video_profile: str
    pixel_format: str
    resolution: MediaResolution
    orientation_degrees: int
    time_base: TimeBase
    frame_pts: tuple[int | None, ...] = Field(min_length=1)
    frame_durations: tuple[int | None, ...] = ()
    duration_seconds: float | None = Field(default=None, gt=0)
    audio_present: bool
    video_stream_index: int = Field(default=0, ge=0)


class MediaRunner(Protocol):
    def tool_versions(self) -> ToolVersions: ...

    def probe(self, source: Path) -> MediaProbe: ...

    def normalize(
        self,
        source: Path,
        destination: Path,
        config: MediaProcessingConfig,
        video_stream_index: int = 0,
    ) -> None: ...

    def frame_hashes(self, source: Path) -> tuple[str, ...]: ...

    def extract_sample_frames(
        self, source: Path, destination: Path, decode_indices: tuple[int, ...]
    ) -> tuple[Path, ...]: ...


def _version_number(output: str, tool: str) -> str:
    first_line = output.splitlines()[0].strip() if output.splitlines() else ""
    match = re.search(rf"^{re.escape(tool)} version\s+([^\s]+)", first_line)
    if not match:
        raise FFmpegVersionError(f"unable to determine {tool} version")
    return match.group(1)


def _library_version(output: str, library: str) -> str:
    match = re.search(rf"^{re.escape(library)}\s+([^\s]+)", output, flags=re.MULTILINE)
    if not match:
        raise FFmpegVersionError(f"unable to determine {library} version")
    return match.group(1)


class SubprocessFFmpegRunner:
    """Run a fixed, audio-free VFR-preserving FFmpeg toolchain."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        timeout_seconds: float | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.timeout_seconds = timeout_seconds

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise FFmpegUnavailableError(
                f"required FFmpeg executable is unavailable: {args[0]}"
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"could not execute {args[0]}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(f"{args[0]} timed out after {exc.timeout} seconds") from exc

    def tool_versions(self) -> ToolVersions:
        ffmpeg_result = self._run([self.ffmpeg, "-version"])
        if ffmpeg_result.returncode != 0:
            raise FFmpegVersionError(
                f"ffmpeg version query failed: {ffmpeg_result.stderr.strip() or 'unknown error'}"
            )
        ffprobe_result = self._run([self.ffprobe, "-version"])
        if ffprobe_result.returncode != 0:
            raise FFmpegVersionError(
                f"ffprobe version query failed: {ffprobe_result.stderr.strip() or 'unknown error'}"
            )
        ffmpeg_version = _version_number(ffmpeg_result.stdout, "ffmpeg")
        ffprobe_version = _version_number(ffprobe_result.stdout, "ffprobe")
        return ToolVersions(
            ffmpeg=ffmpeg_version,
            ffprobe=ffprobe_version,
            decoder=f"libavcodec-{_library_version(ffmpeg_result.stdout, 'libavcodec')}",
        )

    def probe(self, source: Path) -> MediaProbe:
        metadata_result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(source),
            ]
        )
        if metadata_result.returncode != 0:
            _raise_probe_error(metadata_result.stderr)
        try:
            payload = json.loads(metadata_result.stdout)
            video_stream_index, video_stream = next(
                (index, stream)
                for index, stream in enumerate(payload["streams"])
                if stream.get("codec_type") == "video"
                and not stream.get("disposition", {}).get("attached_pic", 0)
            )
            video_ordinal = sum(
                1
                for stream in payload["streams"][:video_stream_index]
                if stream.get("codec_type") == "video"
            )
            numerator_text, denominator_text = video_stream["time_base"].split("/", 1)
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise UnsupportedMediaError(f"unsupported or incomplete media metadata: {exc}") from exc
        frames_result = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_frames",
                "-select_streams",
                f"v:{video_ordinal}",
                "-show_entries",
                "frame=media_type,best_effort_timestamp,pkt_duration",
                str(source),
            ]
        )
        if frames_result.returncode != 0:
            _raise_probe_error(frames_result.stderr)
        try:
            frames_payload = json.loads(frames_result.stdout)
            frame_pts = tuple(
                _optional_int(frame.get("best_effort_timestamp"))
                for frame in frames_payload.get("frames", ())
                if frame.get("media_type", "video") == "video"
            )
            frame_durations = tuple(
                _optional_int(frame.get("pkt_duration"))
                for frame in frames_payload.get("frames", ())
                if frame.get("media_type", "video") == "video"
            )
            if not frame_pts:
                raise KeyError("no decodable frames")
            return MediaProbe(
                container=payload["format"]["format_name"],
                video_codec=video_stream["codec_name"],
                video_profile=video_stream.get("profile") or "unknown",
                pixel_format=video_stream["pix_fmt"],
                resolution=MediaResolution(
                    width_px=video_stream["width"], height_px=video_stream["height"]
                ),
                orientation_degrees=_orientation(video_stream),
                time_base=TimeBase(
                    numerator=int(numerator_text), denominator=int(denominator_text)
                ),
                frame_pts=frame_pts,
                frame_durations=frame_durations,
                duration_seconds=_positive_float(
                    video_stream.get("duration") or payload["format"].get("duration")
                ),
                audio_present=any(
                    stream.get("codec_type") == "audio" for stream in payload["streams"]
                ),
                video_stream_index=video_stream_index,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise UnsupportedMediaError(f"unsupported or incomplete media metadata: {exc}") from exc

    def normalize(
        self,
        source: Path,
        destination: Path,
        config: MediaProcessingConfig,
        video_stream_index: int = 0,
    ) -> None:
        scale = (
            f"scale={config.width_px}:{config.height_px}:force_original_aspect_ratio=decrease,"
            f"pad={config.width_px}:{config.height_px}:(ow-iw)/2:(oh-ih)/2"
        )
        result = self._run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                f"0:{video_stream_index}",
                "-an",
                "-vf",
                scale,
                "-fps_mode",
                "passthrough",
                "-c:v",
                config.encoder,
                "-profile:v",
                config.video_profile.lower(),
                "-pix_fmt",
                config.pixel_format,
                "-threads",
                "1",
                "-map_metadata",
                "-1",
                "-metadata:s:v:0",
                "rotate=0",
                str(destination),
            ]
        )
        if result.returncode != 0 or not destination.is_file():
            raise ToolExecutionError(
                f"FFmpeg normalization failed: {result.stderr.strip() or 'output missing'}"
            )

    def frame_hashes(self, source: Path) -> tuple[str, ...]:
        result = self._run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-f",
                "framehash",
                "-hash",
                "sha256",
                "-",
            ]
        )
        hashes = tuple(
            line.rsplit(",", maxsplit=1)[-1].strip()
            for line in result.stdout.splitlines()
            if line and not line.startswith("#")
        )
        hashes_are_sha256 = all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
        if result.returncode != 0 or not hashes or not hashes_are_sha256:
            raise ToolExecutionError(
                f"FFmpeg frame hashing failed: {result.stderr.strip() or 'no decoded frame hashes'}"
            )
        return hashes

    def extract_sample_frames(
        self, source: Path, destination: Path, decode_indices: tuple[int, ...]
    ) -> tuple[Path, ...]:
        if not decode_indices:
            raise ToolExecutionError("no sample frame indices were requested")
        destination.mkdir()
        selection = "+".join(f"eq(n\\,{index})" for index in decode_indices)
        result = self._run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-vf",
                f"select={selection}",
                "-vsync",
                "0",
                str(destination / "%08d.png"),
            ]
        )
        paths = tuple(sorted(destination.glob("*.png")))
        if result.returncode != 0 or not paths:
            raise ToolExecutionError(
                f"FFmpeg frame decoding failed: {result.stderr.strip() or 'no frames decoded'}"
            )
        return paths


def _optional_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    return int(value)


def _positive_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def _raise_probe_error(stderr: str) -> None:
    reason = stderr.strip() or "ffprobe rejected the input"
    if "invalid data" in reason.lower() or "moov atom" in reason.lower():
        raise CorruptMediaError(reason)
    raise UnsupportedMediaError(reason)


def _orientation(stream: dict) -> int:
    raw = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", ()):
        if "rotation" in side_data:
            raw = side_data["rotation"]
            break
    orientation = int(round(float(raw))) % 360
    if orientation not in {0, 90, 180, 270}:
        raise ValueError(f"unsupported orientation: {orientation}")
    return orientation
