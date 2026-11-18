from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest

from billiards_ai.lab.mainline1_realtime_capture import CaptureFrame
from billiards_ai.lab.mainline1_realtime_ring import FrozenCaptureInterval, MjpegSegmentRing
from billiards_ai.lab.mainline1_recording_media import (
    RecordingPlaybackError,
    load_recording_playback_interval,
    publish_recording_playback,
    verify_recording_playback_artifact,
)
from billiards_ai.lab.mainline1_session_recording import RecordedShotInterval
from billiards_ai.lab.mainline1_shot_jobs import (
    LegacyFrozenJobMediaSourceV1,
    RecordingEndMediaSourceV1,
    ShotMediaEncodeJobV1,
    materialize_frozen_interval,
    materialize_recorded_interval_reference,
)


def _jpeg(frame_id: int) -> bytes:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = frame_id * 20
    cv2.circle(image, (4 + frame_id * 2, 12), 3, (255, 255, 255), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _media_job(
    tmp_path: Path, *, timestamps: dict[int, int] | None = None
) -> tuple[ShotMediaEncodeJobV1, Path]:
    ring = MjpegSegmentRing(
        tmp_path / "raw",
        expected_fps=4,
        pre_roll_frames=0,
        post_roll_frames=0,
        maximum_shot_frames=20,
        maximum_bytes=1024 * 1024,
    )
    for frame_id in range(8):
        ring.append(
            CaptureFrame(
                source_epoch_id="epoch-1",
                capture_frame_id=frame_id,
                captured_at_ns=(timestamps or {}).get(
                    frame_id, round(frame_id * 1_000_000_000 / 60)
                ),
                width=32,
                height=24,
                mjpeg=_jpeg(frame_id),
            )
        )
        if frame_id == 2:
            ring.begin_shot(frame_id)
        if frame_id == 5:
            ring.settle_shot(frame_id)
    frozen = ring.freeze_shot()
    recorded = RecordedShotInterval(
        session_id="session-1",
        shot_id="shot-1",
        root=frozen.root,
        source_epoch_id=frozen.source_epoch_id,
        width=frozen.width,
        height=frozen.height,
        start_frame_id=frozen.start_frame_id,
        end_frame_id=frozen.end_frame_id,
        segment_sha256=dict(frozen.segment_sha256),
    )
    return _job_from_recorded_interval(recorded, tmp_path / "reference"), frozen.root


def _job_from_recorded_interval(recorded, reference_root):
    materialize_recorded_interval_reference(recorded, reference_root, recording_identity=True)
    reference = reference_root / "interval.json"
    job = ShotMediaEncodeJobV1.create(
        session_id=recorded.session_id,
        shot_id=recorded.shot_id,
        source_epoch_id=recorded.source_epoch_id,
        start_frame_id=recorded.start_frame_id,
        end_frame_id=recorded.end_frame_id,
        media_root=reference_root,
        media_sha256=recorded.media_sha256,
        source=RecordingEndMediaSourceV1(
            kind="recording_end",
            recording_end_sha256="a" * 64,
            shot_evidence_sha256="b" * 64,
            reference_path=reference,
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        ),
    )
    return job


def test_larger_interval_encodes_without_all_frame_materialization(tmp_path, monkeypatch):
    count, width, height = 360, 640, 480
    ring = MjpegSegmentRing(
        tmp_path / "raw",
        expected_fps=60,
        pre_roll_frames=0,
        post_roll_frames=0,
        maximum_shot_frames=count + 1,
        maximum_bytes=64 * 1024**2,
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    for frame_id in range(count):
        ring.append(
            CaptureFrame("epoch-large", frame_id, round(frame_id * 1e9 / 60), width, height, jpeg)
        )
        if frame_id == 0:
            ring.begin_shot(frame_id)
        if frame_id == count - 1:
            ring.settle_shot(frame_id)
    frozen = ring.freeze_shot()
    recorded = RecordedShotInterval(
        session_id="session-large",
        shot_id="shot-large",
        root=frozen.root,
        source_epoch_id=frozen.source_epoch_id,
        width=width,
        height=height,
        start_frame_id=frozen.start_frame_id,
        end_frame_id=frozen.end_frame_id,
        segment_sha256=dict(frozen.segment_sha256),
    )
    job = _job_from_recorded_interval(recorded, tmp_path / "reference")
    original_frames = FrozenCaptureInterval.frames
    yielded = 0

    class StreamingOnly:
        def __init__(self, iterator):
            self.iterator = iterator

        def __iter__(self):
            nonlocal yielded
            for frame in self.iterator:
                yielded += 1
                yield frame

        def __length_hint__(self):
            raise AssertionError(
                "media path attempted tuple/list materialization of original frames"
            )

    for materialize in (list, tuple):
        with pytest.raises(AssertionError, match="materialization"):
            materialize(StreamingOnly(iter(())))
    monkeypatch.setattr(
        FrozenCaptureInterval,
        "frames",
        lambda self, **kwargs: StreamingOnly(original_frames(self, **kwargs)),
    )
    tracemalloc.start()
    try:
        artifact = publish_recording_playback(
            job, tmp_path / "output", authority_check=lambda _: None
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert artifact.frame_count == count
    assert artifact.width == width and artifact.height == height
    assert yielded == count * 2  # bounded timing pass and one decode/encode pass
    # A decoded all-frame batch would exceed 316 MiB for this input.
    # This measures Python/NumPy allocations, not native codec RSS or throughput.
    assert peak < 48 * 1024**2
    print(f"streaming_probe frames={count} geometry={width}x{height} peak_python_bytes={peak}")


def test_real_original_frames_publish_as_verified_h264_artifact(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)
    authority_checks = []

    artifact = publish_recording_playback(
        job,
        tmp_path / "artifacts",
        authority_check=lambda checked: authority_checks.append(checked.job_id),
    )
    repeated = publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )

    assert artifact == repeated
    assert artifact.frame_count == job.end_frame_id - job.start_frame_id + 1
    assert artifact.width == 32 and artifact.height == 24
    assert artifact.fps == pytest.approx(60, abs=0.01)
    capture = cv2.VideoCapture(str(artifact.media_path))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    assert "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).lower() in {
        "avc1",
        "h264",
        "x264",
    }
    assert verify_recording_playback_artifact(artifact.path, job) == artifact
    assert list((tmp_path / "artifacts").glob(".pending-*")) == []
    assert len(authority_checks) == 2


@pytest.mark.parametrize("stop_after", [1, 3])
def test_artifact_verification_yields_during_hash_and_decode(tmp_path, stop_after):
    job, _ = _media_job(tmp_path)
    artifact = publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    calls = 0

    def preempt():
        nonlocal calls
        calls += 1
        if calls == stop_after:
            raise RecordingPlaybackError("MEDIA_LIVE_PRIORITY", "next match started")

    with pytest.raises(RecordingPlaybackError) as failure:
        verify_recording_playback_artifact(artifact.path, job, continue_check=preempt)
    assert failure.value.code == "MEDIA_LIVE_PRIORITY"
    assert calls == stop_after


def test_unavailable_avc_encoder_is_an_honest_typed_failure(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)

    class ClosedWriter:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(
            job,
            tmp_path / "artifacts",
            authority_check=lambda _: None,
            writer_factory=ClosedWriter,
        )

    assert error.value.code == "MEDIA_H264_ENCODER_UNAVAILABLE"
    assert not list((tmp_path / "artifacts").glob(".pending-*"))


def test_reference_hash_mutation_fails_before_publication(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)
    job.source.reference_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(job, tmp_path / "artifacts", authority_check=lambda _: None)

    assert error.value.code == "MEDIA_REFERENCE_HASH_MISMATCH"


def test_original_segment_mutation_fails_before_availability(tmp_path: Path) -> None:
    job, raw_root = _media_job(tmp_path)
    segment = next(path for path in raw_root.iterdir() if path.suffix == ".mjpgseg")
    segment.write_bytes(segment.read_bytes() + b"changed")

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(job, tmp_path / "artifacts", authority_check=lambda _: None)

    assert error.value.code == "MEDIA_SOURCE_INVALID"


def test_segment_mutation_between_timing_and_encode_stays_source_invalid(
    tmp_path: Path,
) -> None:
    job, raw_root = _media_job(tmp_path)
    segment = next(path for path in raw_root.iterdir() if path.suffix == ".mjpgseg")
    checks = 0

    def mutate_on_second_pass():
        nonlocal checks
        checks += 1
        if checks == 7:
            segment.write_bytes(segment.read_bytes() + b"changed")

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(
            job,
            tmp_path / "artifacts",
            authority_check=lambda _: None,
            continue_check=mutate_on_second_pass,
        )
    assert error.value.code == "MEDIA_SOURCE_INVALID"


def test_published_media_tampering_is_rejected(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)
    artifact = publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    artifact.media_path.write_bytes(artifact.media_path.read_bytes() + b"changed")

    with pytest.raises(RecordingPlaybackError):
        verify_recording_playback_artifact(artifact.path, job)


def test_legacy_source_recomputes_interval_identity(tmp_path: Path) -> None:
    recording_job, _ = _media_job(tmp_path)
    frozen = load_recording_playback_interval(recording_job)
    legacy_root = tmp_path / "legacy-reference"
    materialize_frozen_interval(frozen, legacy_root)
    legacy_job = ShotMediaEncodeJobV1.create(
        session_id=recording_job.session_id,
        shot_id=recording_job.shot_id,
        source_epoch_id=recording_job.source_epoch_id,
        start_frame_id=recording_job.start_frame_id,
        end_frame_id=recording_job.end_frame_id,
        media_root=legacy_root,
        media_sha256=recording_job.media_sha256,
        source=LegacyFrozenJobMediaSourceV1(
            kind="legacy_frozen_job",
            postgame_manifest_sha256="c" * 64,
            source_job_id="analysis-job-1",
            source_job_payload_sha256="d" * 64,
        ),
    )

    artifact = publish_recording_playback(
        legacy_job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    assert artifact.frame_count == legacy_job.end_frame_id - legacy_job.start_frame_id + 1
    reference = legacy_root / "interval.json"
    payload = json.loads(reference.read_text())
    first_name = next(iter(payload["segment_sha256"]))
    payload["segment_sha256"] = {first_name: payload["segment_sha256"][first_name]}
    reference.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(
            legacy_job, tmp_path / "other-artifacts", authority_check=lambda _: None
        )
    assert error.value.code == "MEDIA_REFERENCE_LINEAGE_INVALID"


def test_discontinuous_capture_cadence_is_rejected(tmp_path: Path) -> None:
    timestamps = {
        0: 0,
        1: 16_666_667,
        2: 33_333_334,
        3: 50_000_001,
        4: 550_000_001,
        5: 566_666_668,
        6: 583_333_335,
        7: 600_000_002,
    }
    job, _ = _media_job(tmp_path, timestamps=timestamps)
    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(job, tmp_path / "artifacts", authority_check=lambda _: None)
    assert error.value.code == "MEDIA_TIMING_INVALID"


def test_authority_is_rechecked_immediately_before_publication(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)
    checks = 0

    def authority(_):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RecordingPlaybackError("MEDIA_AUTHORITY_REVOKED", "shot was discarded")

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(job, tmp_path / "artifacts", authority_check=authority)
    assert error.value.code == "MEDIA_AUTHORITY_REVOKED"
    assert not list((tmp_path / "artifacts").iterdir())


def test_non_avc_video_with_avc1_text_is_rejected(tmp_path: Path) -> None:
    job, _ = _media_job(tmp_path)

    class Mp4vWriter:
        def __init__(self, path, _fourcc, fps, geometry):
            self.delegate = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, geometry)

        def isOpened(self):
            return self.delegate.isOpened()

        def write(self, image):
            return self.delegate.write(image)

        def release(self):
            self.delegate.release()

    with pytest.raises(RecordingPlaybackError) as error:
        publish_recording_playback(
            job,
            tmp_path / "artifacts",
            authority_check=lambda _: None,
            writer_factory=Mp4vWriter,
        )
    assert error.value.code == "MEDIA_CODEC_INVALID"


def test_destination_race_accepts_only_the_verified_winner(tmp_path: Path, monkeypatch) -> None:
    job, _ = _media_job(tmp_path)
    import billiards_ai.lab.mainline1_recording_media as media

    real_rename = media.os.rename

    def raced_rename(source, destination):
        real_rename(source, destination)
        raise FileExistsError("simulated concurrent winner")

    monkeypatch.setattr(media.os, "rename", raced_rename)
    artifact = publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    assert verify_recording_playback_artifact(artifact.path, job) == artifact


def test_real_playback_publication_flushes_with_windows_write_access(tmp_path, monkeypatch):
    """Model only the documented Windows flush access rule on a real descriptor."""
    import os
    import stat

    import billiards_ai.lab.mainline1_recording_media as media

    job, _ = _media_job(tmp_path)
    native_fsync = os.fsync
    flushed = []

    def windows_flush(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            if os.name == "posix":
                import errno
                import fcntl

                flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
                if flags & os.O_ACCMODE == os.O_RDONLY:
                    raise OSError(errno.EBADF, "Windows flush requires write access")
            flushed.append(descriptor)
        native_fsync(descriptor)

    monkeypatch.setattr(media.os, "fsync", windows_flush)
    artifact = publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    verified = verify_recording_playback_artifact(artifact.path, job)
    assert flushed and verified.content_sha256 == artifact.content_sha256
    assert verified.frame_count == job.end_frame_id - job.start_frame_id + 1
    assert not tuple((tmp_path / "artifacts").glob(".pending-recording-playback-*"))


def test_output_flush_does_not_create_a_missing_encoded_file(tmp_path):
    import billiards_ai.lab.mainline1_recording_media as media

    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError):
        media._fsync_file(missing)
    assert not missing.exists()


def test_new_controlled_mp4_is_fully_decoded_once_before_publication(tmp_path, monkeypatch):
    import billiards_ai.lab.mainline1_recording_media as media

    job, _ = _media_job(tmp_path)
    checks = []
    original = media._verify_mp4

    def verify(path, **kwargs):
        checks.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(media, "_verify_mp4", verify)
    artifact = media.publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    assert artifact.media_path.is_file()
    assert len(checks) == 1
    # Recovery has no in-process proof and still fully revalidates the existing file.
    media.verify_recording_playback_artifact(artifact.path, job)
    assert len(checks) == 2


@pytest.mark.parametrize("change", ["same-inode-content", "different-file-same-bytes"])
def test_generated_receipt_never_trusts_weak_stat_identity_or_foreign_file(
    tmp_path, monkeypatch, change
):
    import os

    import billiards_ai.lab.mainline1_recording_media as media

    job, _ = _media_job(tmp_path)
    original_rename, original_verify = media.os.rename, media._verify_mp4
    checks = []

    def verify(path, **kwargs):
        checks.append(path)
        return original_verify(path, **kwargs)

    def rename(source, destination):
        original_rename(source, destination)
        path = destination / "original.mp4"
        old = path.stat()
        data = path.read_bytes()
        if change == "same-inode-content":
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        else:
            other = destination / "replacement.tmp"
            other.write_bytes(data)
            os.replace(other, path)
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))

    monkeypatch.setattr(media.os, "rename", rename)
    monkeypatch.setattr(media, "_verify_mp4", verify)
    if change == "same-inode-content":
        with pytest.raises(RecordingPlaybackError):
            media.publish_recording_playback(
                job, tmp_path / "artifacts", authority_check=lambda _: None
            )
    else:
        assert media.publish_recording_playback(
            job, tmp_path / "artifacts", authority_check=lambda _: None
        )
    assert len(checks) == 2


def test_recovery_fully_validates_existing_output_without_reencoding(tmp_path, monkeypatch):
    import billiards_ai.lab.mainline1_recording_media as media

    job, _ = _media_job(tmp_path)
    artifact = media.publish_recording_playback(
        job, tmp_path / "artifacts", authority_check=lambda _: None
    )
    checks = []
    verify = media._verify_mp4

    def decoded(path, **kwargs):
        checks.append(path)
        return verify(path, **kwargs)

    monkeypatch.setattr(media, "_verify_mp4", decoded)
    monkeypatch.setattr(
        media,
        "_encode_interval",
        lambda *a, **k: pytest.fail("recovery reencoded a verified output"),
    )
    assert (
        media.publish_recording_playback(
            job, tmp_path / "artifacts", authority_check=lambda _: None
        )
        == artifact
    )
    assert len(checks) == 1


def test_replacement_immediately_after_decode_cannot_receive_generated_proof(tmp_path, monkeypatch):
    import os

    import billiards_ai.lab.mainline1_recording_media as media

    job, _ = _media_job(tmp_path)
    original = media._verify_mp4

    def replace_after_decode(path, **kwargs):
        verified = original(path, **kwargs)
        other = path.with_suffix(".replacement")
        other.write_bytes(path.read_bytes())
        os.replace(other, path)
        return verified

    monkeypatch.setattr(media, "_verify_mp4", replace_after_decode)
    with pytest.raises(media.RecordingPlaybackError, match="changed during validation"):
        media.publish_recording_playback(
            job, tmp_path / "artifacts", authority_check=lambda _: None
        )
