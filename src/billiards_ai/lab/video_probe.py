"""Deterministic video metadata normalization and sequential frame sampling."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from pydantic import Field

from billiards_ai.lab.contracts import FrozenContract, NonEmptyStr


class VideoProbeError(RuntimeError):
    """A video cannot be probed or deterministically decoded."""


class FfprobeRunner(Protocol):
    def run(self, path: Path) -> dict[str, object]: ...


class SubprocessFfprobeRunner:
    """Run the local ffprobe binary and return its JSON document."""

    def run(self, path: Path) -> dict[str, object]:
        command = (
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            "-show_frames",
            "-show_entries",
            (
                "stream=index,codec_type,codec_name,profile,pix_fmt,width,height,"
                "r_frame_rate,avg_frame_rate,time_base,nb_frames,duration:"
                "stream_tags=rotate:stream_side_data=rotation:"
                "format=format_name,duration:frame=media_type,best_effort_timestamp_time"
            ),
            str(path),
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit {completed.returncode}"
            raise VideoProbeError(f"ffprobe failed for {path}: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise VideoProbeError(f"ffprobe returned invalid JSON for {path}") from exc
        if not isinstance(payload, dict):
            raise VideoProbeError(f"ffprobe returned an invalid document for {path}")
        return payload


class VideoProbe(FrozenContract):
    path: NonEmptyStr
    container: NonEmptyStr
    codec: NonEmptyStr
    profile: str | None = None
    pixel_format: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    nominal_fps: float = Field(ge=0)
    average_fps: float = Field(ge=0)
    time_base: NonEmptyStr
    frame_count: int | None = Field(default=None, ge=0)
    duration_seconds: float = Field(ge=0)
    audio_stream_count: int = Field(ge=0)
    timestamps_monotonic: bool
    rotation_degrees: int = 0


@dataclass(frozen=True)
class FrameQuality:
    luma_mean: float
    luma_p05: float
    luma_p95: float
    clipped_white_ratio: float
    laplacian_variance: float


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_ms: float
    image: np.ndarray
    quality: FrameQuality


def _rate(value: object) -> float:
    if value in {None, "", "0/0"}:
        return 0.0
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoProbeError(f"invalid frame rate: {value}") from exc


def _rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        return int(float(str(tags["rotate"]))) % 360
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                return int(float(str(item["rotation"]))) % 360
    return 0


def probe_video(path: Path, ffprobe: FfprobeRunner | None = None) -> VideoProbe:
    """Normalize ffprobe output while treating decoded timestamps as authoritative."""

    payload = (ffprobe or SubprocessFfprobeRunner()).run(path)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoProbeError("ffprobe document is missing streams")
    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    if not video_streams:
        raise VideoProbeError("video stream is missing")
    stream = video_streams[0]
    audio_count = sum(
        isinstance(candidate, dict) and candidate.get("codec_type") == "audio"
        for candidate in streams
    )
    frames = payload.get("frames")
    timestamps: list[float] = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            if frame.get("media_type") not in {None, "video"}:
                continue
            value = frame.get("best_effort_timestamp_time")
            if value not in {None, "N/A"}:
                timestamps.append(float(str(value)))
    monotonic = all(
        current >= previous for previous, current in zip(timestamps, timestamps[1:], strict=False)
    )
    format_payload = payload.get("format")
    format_data = format_payload if isinstance(format_payload, dict) else {}
    duration_value = stream.get("duration", format_data.get("duration", 0))
    frame_count_value = stream.get("nb_frames")
    frame_count = None if frame_count_value in {None, "N/A"} else int(str(frame_count_value))

    return VideoProbe(
        path=str(path),
        container=str(format_data.get("format_name") or "unknown"),
        codec=str(stream.get("codec_name") or "unknown"),
        profile=None if stream.get("profile") is None else str(stream["profile"]),
        pixel_format=None if stream.get("pix_fmt") is None else str(stream["pix_fmt"]),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        nominal_fps=_rate(stream.get("r_frame_rate")),
        average_fps=_rate(stream.get("avg_frame_rate")),
        time_base=str(stream.get("time_base") or "unknown"),
        frame_count=frame_count,
        duration_seconds=float(str(duration_value or 0)),
        audio_stream_count=int(audio_count),
        timestamps_monotonic=monotonic,
        rotation_degrees=_rotation(stream),
    )


def _frame_quality(image: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return FrameQuality(
        luma_mean=float(gray.mean()),
        luma_p05=float(np.percentile(gray, 5)),
        luma_p95=float(np.percentile(gray, 95)),
        clipped_white_ratio=float(np.count_nonzero(gray >= 250) / gray.size),
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def sample_frames(path: Path, frame_indices: tuple[int, ...]) -> tuple[SampledFrame, ...]:
    """Decode from frame zero and return exact requested indices without seek approximation."""

    if any(index < 0 for index in frame_indices):
        raise ValueError("frame indices must be non-negative")
    if tuple(sorted(set(frame_indices))) != frame_indices:
        raise ValueError("frame indices must be strictly increasing and unique")
    if not frame_indices:
        return ()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoProbeError(f"video cannot be opened: {path}")
    requested = set(frame_indices)
    sampled: list[SampledFrame] = []
    frame_index = 0
    maximum = frame_indices[-1]
    try:
        while frame_index <= maximum:
            ok, image = capture.read()
            if not ok:
                break
            if frame_index in requested:
                sampled.append(
                    SampledFrame(
                        frame_index=frame_index,
                        timestamp_ms=float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                        image=image.copy(),
                        quality=_frame_quality(image),
                    )
                )
            frame_index += 1
    finally:
        capture.release()
    found = {frame.frame_index for frame in sampled}
    missing = [index for index in frame_indices if index not in found]
    if missing:
        raise VideoProbeError(f"requested frame was not decoded: {missing[0]}")
    return tuple(sampled)
