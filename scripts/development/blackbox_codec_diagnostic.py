"""Compare AUTO and forced FFmpeg on the same 40 real frames; never product acceptance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

HELPER_SHA = "a51686eb6dbe046a057ae074b09fe7644823e9d79890e7c29d2d45e2751b990d"
helper_path = Path(__file__).with_name("blackbox_media_diagnostic.py")
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("HELPER_IDENTITY")
spec = importlib.util.spec_from_file_location("bounded_media_component_helper", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
BASE_EXPORT = helper.export_events
BACKENDS = {"FFMPEG", "MSMF", "AVFOUNDATION", "GSTREAMER", "UNAVAILABLE", "OTHER"}
VARIANTS = {"AUTO", "FORCED_FFMPEG"}
CODEC_PIN = {
    "bytes": 986712,
    "sha256": "bf91c488638b32448f7d89611f6bb9c232dce82975d7d2827c78f78fc2a7fb2b",
}


def preload_codec(path, profile, *, loader=None):
    """Load only the already-pinned DLL by full path in this fresh diagnostic process."""
    profile.check()
    helper.safe_under(path, path.parent)
    if path.stat().st_size != CODEC_PIN["bytes"]:
        raise ValueError("CODEC_IDENTITY")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CODEC_PIN["sha256"]:
            raise ValueError("CODEC_IDENTITY")
    if loader is None:
        import ctypes

        loader = ctypes.WinDLL
    try:
        handle = loader(str(path))
    except OSError as exc:
        code = getattr(exc, "winerror", None)
        profile.snapshot(
            "CODEC_PRELOAD",
            opened=False,
            winerror=code if type(code) is int and 0 <= code < 2**31 else 0,
        )
        return None
    profile.snapshot("CODEC_PRELOAD", opened=True)
    return handle


def export_events(raw):
    result = []
    for line in raw.splitlines()[:256]:
        clean = BASE_EXPORT(line)
        if not clean:
            continue
        row = json.loads(line)
        item = clean[0]
        for key, allowed in (("variant", VARIANTS), ("backend", BACKENDS)):
            if isinstance(row.get(key), str) and row[key] in allowed:
                item[key] = row[key]
        for key in ("fps", "width", "height", "source_frames", "winerror"):
            value = row.get(key)
            if type(value) in {int, float} and math.isfinite(value) and 0 <= value < 2**63:
                item[key] = value
        for key in ("opened", "verified"):
            if type(row.get(key)) is bool:
                item[key] = row[key]
        result.append(item)
    return result


def encode_pair(interval, output, fps, profile, *, writer_factory=None):
    import cv2
    import numpy as np

    factory = cv2.VideoWriter if writer_factory is None else writer_factory
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writers, results = [], []
    primary = None
    frames = None
    try:
        for variant in ("AUTO", "FORCED_FFMPEG"):
            target = output / (variant.lower() + ".mp4")
            args = (str(target), fourcc, fps, (interval.width, interval.height))
            if variant == "FORCED_FFMPEG":
                args = (str(target), cv2.CAP_FFMPEG, *args[1:])
            writer = profile.wrap(variant.lower() + "_open", factory)(*args)
            # Register before any backend query so exceptions still release this exact writer.
            writers.append((variant, writer))
            opened = bool(writer.isOpened())
            backend = writer.getBackendName() if opened else "UNAVAILABLE"
            backend = backend if backend in BACKENDS else "OTHER"
            result = dict(variant=variant, backend=backend, opened=opened, frames=0)
            results.append(result)
            profile.snapshot(
                "CODEC_WRITER_OPEN", **result, fps=fps, width=interval.width, height=interval.height
            )
        if not any(row["opened"] for row in results):
            raise ValueError("NO_ENCODER")
        frames = interval.frames(before_segment=profile.check)
        first = None
        for index in range(40):
            profile.check()
            frame = next(frames)
            if first is None:
                first = frame.capture_frame_id
            if frame.capture_frame_id != first + index:
                raise ValueError("FRAME_SEQUENCE")
            image = profile.wrap("sample_jpeg_decode", cv2.imdecode)(
                np.frombuffer(frame.mjpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None or image.shape != (interval.height, interval.width, 3):
                raise ValueError("FRAME_GEOMETRY")
            for (variant, writer), result in zip(writers, results, strict=True):
                if result["opened"]:
                    profile.wrap(variant.lower() + "_write", writer.write)(image)
                    result["frames"] += 1
            if (index + 1) % 10 == 0:
                profile.snapshot("CODEC_FRAME_PROGRESS", frames=index + 1)
        return results
    except BaseException as exc:
        primary = exc
        raise
    finally:
        release_error = None
        if frames is not None:
            try:
                frames.close()
            except BaseException as exc:
                release_error = exc
        for variant, writer in writers:
            try:
                profile.wrap(variant.lower() + "_release", writer.release, check_budget=False)()
            except BaseException as exc:
                release_error = release_error or exc
        if release_error is not None:
            if primary is not None:
                primary.add_note("codec writer release also failed")
            else:
                raise release_error


def profile_codecs(recording, output, profile, *, verify_imports=False):
    codec_handle = None
    if sys.platform == "win32":
        codec_handle = preload_codec(recording.parent / "codec/openh264-2.5.0-win64.dll", profile)
    import cv2
    import numpy

    from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
    from billiards_ai.lab import mainline1_recording_media as media
    from billiards_ai.lab.mainline1_shot_jobs import SQLiteShotAnalysisJobStore
    from billiards_ai.lab.mainline1_shot_worker import Mainline1RecordingMediaWorker

    if verify_imports:
        run = recording.parent
        for module, root in (
            (media, run / "source/src"),
            (cv2, run / "dependencies"),
            (numpy, run / "dependencies"),
        ):
            helper.safe_under(Path(module.__file__), root)
    for source, target in (
        (recording / "state.sqlite3", output / "state.sqlite3"),
        (recording / "jobs/analysis-jobs.sqlite3", output / "jobs.sqlite3"),
    ):
        helper.safe_under(source, recording)
        helper.backup_database(source, target, profile.check)
    records = SQLiteFieldRecordStore(output / "state.sqlite3")
    jobs = SQLiteShotAnalysisJobStore(output / "jobs.sqlite3")
    with closing(sqlite3.connect(output / "jobs.sqlite3")) as connection:
        sessions = connection.execute(
            "SELECT DISTINCT json_extract(payload_json,'$.session_id') "
            "FROM shot_analysis_jobs WHERE job_kind='media' LIMIT 2"
        ).fetchall()
    if len(sessions) != 1:
        raise ValueError("INVENTORY")
    items = jobs.list_session_media_jobs(sessions[0][0])
    if len(items) != 28:
        raise ValueError("INVENTORY")
    job = items[0]
    helper.safe_under(job.media_root, recording)
    helper.safe_under(job.source.reference_path, recording)
    if job.source.reference_path.stat().st_size > 1024**2:
        raise ValueError("REFERENCE_SIZE")
    reference = json.loads(job.source.reference_path.read_bytes())
    helper.safe_under(Path(reference["segment_root"]), recording)
    worker = Mainline1RecordingMediaWorker(
        records=records, jobs=jobs, artifact_root=output / "artifacts"
    )
    worker._validate_authority(job)
    interval = media.load_recording_playback_interval(job)
    if interval.frame_count < 40 or interval.frame_count > 1200:
        raise ValueError("FRAME_BUDGET")
    timing = profile.call("MEASURE_TIMING", media._measure_timing, interval, profile.check)
    profile.snapshot(
        "CODEC_SAMPLE",
        frames=40,
        source_frames=interval.frame_count,
        fps=timing["fps"],
        width=interval.width,
        height=interval.height,
    )
    rows = encode_pair(interval, output, timing["fps"], profile)
    for row in rows:
        if row["opened"]:
            path = output / (row["variant"].lower() + ".mp4")
            info = profile.call(
                "VERIFY_MP4",
                media._verify_mp4,
                path,
                expected_frames=40,
                width=interval.width,
                height=interval.height,
                fps=timing["fps"],
                continue_check=profile.check,
            )
            profile.snapshot("CODEC_OUTPUT", **row, verified=True, output_bytes=info["byte_size"])
    worker._validate_authority(job)
    if sys.platform == "win32":
        # Keep the native module referenced until both writers and verification have finished.
        profile.snapshot("CODEC_PRELOAD_RETAINED", opened=codec_handle is not None)
    profile.snapshot("COMPLETE")


# Reuse the already-reviewed ownership, immutable source/runtime checks, fresh directories,
# 55/60/10-second budgets, 128MiB per-file budget and three-file finite export.
helper.CONTRACT = "blackbox-media-codec-startupenv/v1"
helper.EVENT_NAMES |= {
    "CODEC_WRITER_OPEN",
    "CODEC_FRAME_PROGRESS",
    "CODEC_SAMPLE",
    "CODEC_OUTPUT",
    "CODEC_PRELOAD",
    "CODEC_PRELOAD_RETAINED",
}
helper.METRIC_NAMES |= {
    "sample_jpeg_decode",
    "auto_open",
    "auto_write",
    "auto_release",
    "forced_ffmpeg_open",
    "forced_ffmpeg_write",
    "forced_ffmpeg_release",
}
helper.profile_components = profile_codecs
helper.export_events = export_events
helper.__file__ = str(Path(__file__).resolve())

if __name__ == "__main__":
    raise SystemExit(helper.main())
