"""Deterministic VFR media normalization with all-or-nothing publication."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from billiards_ai.shared.media_timing import select_frame_samples
from billiards_ai.shared.models.calibration import StrictFrozenModel
from billiards_ai.shared.models.media import (
    FrameManifest,
    FrameRateStrategy,
    FrameRecord,
    MediaEquivalenceRule,
    MediaResolution,
    NormalizedMedia,
    frame_sequence_sha256,
)

from .errors import (
    MediaIntegrityError,
    MediaProcessingError,
    MediaPublicationError,
    ResourceLimitError,
)
from .runner import MediaProcessingConfig, MediaRunner

LOGGER = logging.getLogger(__name__)
NORMALIZED_MEDIA_FILE = "normalized-media.json"
FRAME_MANIFEST_FILE = "frame-manifest.json"


class MediaProcessingResult(StrictFrozenModel):
    normalized_media: NormalizedMedia
    frame_manifest: FrameManifest


class MediaProcessingLimits(StrictFrozenModel):
    """Caller-selected bounds; deployment policy, not product defaults."""

    max_input_bytes: int = Field(gt=0)
    max_decoded_frames: int = Field(gt=0)
    max_sampled_frames: int = Field(gt=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: MediaProcessingConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def _write_model(path: Path, model: BaseModel) -> None:
    payload = model.model_dump_json(indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class MediaProcessingService:
    def __init__(self, runner: MediaRunner, *, limits: MediaProcessingLimits) -> None:
        self.runner = runner
        self.limits = limits

    def process(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        config: MediaProcessingConfig,
    ) -> MediaProcessingResult:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        if not input_path.is_file():
            raise MediaPublicationError(f"input media is not a file: {input_path}")
        if input_path.stat().st_size > self.limits.max_input_bytes:
            raise ResourceLimitError("input size exceeds the configured processing limit")
        if output_dir.exists():
            raise MediaPublicationError(f"publication target already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
        )
        try:
            result = self._build(input_path=input_path, staging=staging, config=config)
            verified = verify_published_media(staging)
            if verified != result:
                raise MediaIntegrityError("staged contracts changed before publication")
            os.rename(staging, output_dir)
            return result
        except MediaProcessingError as exc:
            LOGGER.error("media processing failed: %s", exc)
            raise
        except Exception as exc:
            LOGGER.error("media processing failed: %s", exc)
            raise MediaPublicationError(f"media publication failed: {exc}") from exc
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _build(
        self,
        *,
        input_path: Path,
        staging: Path,
        config: MediaProcessingConfig,
    ) -> MediaProcessingResult:
        tool_versions = self.runner.tool_versions()
        probe = self.runner.probe(input_path)
        if len(probe.frame_pts) > self.limits.max_decoded_frames:
            raise ResourceLimitError("decoded frame count exceeds the configured processing limit")
        samples = select_frame_samples(
            probe.frame_pts,
            time_base=probe.time_base,
            interval_ms=config.interval_ms,
        )
        if len(samples) > self.limits.max_sampled_frames:
            raise ResourceLimitError("sampled frame count exceeds the configured processing limit")
        normalized_path = staging / "normalized.mp4"
        self.runner.normalize(input_path, normalized_path, config, probe.video_stream_index)
        normalized_probe = self.runner.probe(normalized_path)
        _validate_normalized_probe(normalized_probe, config)
        decoded_hashes = self.runner.frame_hashes(normalized_path)
        if len(decoded_hashes) != len(probe.frame_pts):
            raise MediaIntegrityError(
                "decoded frame hash count does not match source PTS evidence "
                f"({len(decoded_hashes)} != {len(probe.frame_pts)})"
            )
        normalized_equivalence_hash = frame_sequence_sha256(decoded_hashes)

        frames_dir = staging / "frames"
        frames_dir.mkdir()
        unique_decode_indices = tuple(dict.fromkeys(sample.decode_index for sample in samples))
        extracted_paths = self.runner.extract_sample_frames(
            normalized_path, staging / "sampled", unique_decode_indices
        )
        if len(extracted_paths) != len(unique_decode_indices):
            raise MediaIntegrityError(
                "sample extraction did not return every requested decoded frame"
            )
        extracted_by_decode_index = dict(zip(unique_decode_indices, extracted_paths, strict=True))
        records: list[FrameRecord] = []
        for index, sample in enumerate(samples):
            destination = frames_dir / f"{index:06d}.png"
            shutil.copyfile(extracted_by_decode_index[sample.decode_index], destination)
            records.append(
                FrameRecord(
                    index=index,
                    requested_timestamp_ms=sample.requested_timestamp_ms,
                    source_pts=sample.source_pts,
                    image_path=destination.relative_to(staging).as_posix(),
                    content_sha256=_sha256(destination),
                )
            )
        shutil.rmtree(staging / "sampled")

        input_hash = _sha256(input_path)
        config_hash = _config_sha256(config)
        manifest_equivalence_hash = frame_sequence_sha256(
            tuple(record.content_sha256 for record in records)
        )
        equivalence_rule = MediaEquivalenceRule.DECODED_FRAME_SHA256_V1
        manifest = FrameManifest(
            input_sha256=input_hash,
            processing_config_sha256=config_hash,
            interval_ms=config.interval_ms,
            source_time_base=probe.time_base,
            first_source_pts=next(pts for pts in probe.frame_pts if pts is not None),
            tool_versions=tool_versions,
            equivalence_rule=equivalence_rule,
            decoded_frames_sha256=manifest_equivalence_hash,
            frames=tuple(records),
        )
        normalized = NormalizedMedia(
            input_sha256=input_hash,
            processing_config_sha256=config_hash,
            media_path=normalized_path.relative_to(staging).as_posix(),
            container=config.container,
            video_codec=config.video_codec,
            video_profile=config.video_profile,
            pixel_format=config.pixel_format,
            resolution=normalized_probe.resolution,
            source_orientation_degrees=probe.orientation_degrees,
            orientation_degrees=normalized_probe.orientation_degrees,
            time_base=normalized_probe.time_base,
            frame_rate_strategy=FrameRateStrategy.PRESERVE_VFR_SAMPLE_BY_PTS,
            audio_removed=True,
            tool_versions=tool_versions,
            content_sha256=_sha256(normalized_path),
            equivalence_rule=equivalence_rule,
            decoded_frames_sha256=normalized_equivalence_hash,
        )
        result = MediaProcessingResult(normalized_media=normalized, frame_manifest=manifest)
        _write_model(staging / NORMALIZED_MEDIA_FILE, normalized)
        _write_model(staging / FRAME_MANIFEST_FILE, manifest)
        return result


def _validate_normalized_probe(probe, config: MediaProcessingConfig) -> None:
    expected_resolution = MediaResolution(width_px=config.width_px, height_px=config.height_px)
    if config.container not in probe.container.split(","):
        raise MediaIntegrityError(f"normalized container does not match {config.container}")
    if probe.video_codec != config.video_codec:
        raise MediaIntegrityError(f"normalized video codec does not match {config.video_codec}")
    if probe.video_profile.casefold() != config.video_profile.casefold():
        raise MediaIntegrityError(f"normalized video profile does not match {config.video_profile}")
    if probe.pixel_format != config.pixel_format:
        raise MediaIntegrityError(f"normalized pixel format does not match {config.pixel_format}")
    if probe.resolution != expected_resolution:
        raise MediaIntegrityError("normalized resolution does not match processing config")
    if probe.orientation_degrees != 0:
        raise MediaIntegrityError("normalized orientation was not cleared")
    if probe.audio_present:
        raise MediaIntegrityError("normalized media still contains audio")


def verify_published_media(output_dir: Path) -> MediaProcessingResult:
    """Verify publication files and sampled-frame evidence (not full re-decoding)."""

    output_dir = Path(output_dir)
    try:
        normalized = NormalizedMedia.model_validate_json(
            (output_dir / NORMALIZED_MEDIA_FILE).read_text(encoding="utf-8")
        )
        manifest = FrameManifest.model_validate_json(
            (output_dir / FRAME_MANIFEST_FILE).read_text(encoding="utf-8")
        )
        if _sha256(output_dir / normalized.media_path) != normalized.content_sha256:
            raise MediaIntegrityError("normalized media content hash mismatch")
        actual_frame_hashes = tuple(
            _sha256(output_dir / frame.image_path) for frame in manifest.frames
        )
        if actual_frame_hashes != tuple(frame.content_sha256 for frame in manifest.frames):
            raise MediaIntegrityError("frame content hash mismatch")
        actual_equivalence = frame_sequence_sha256(actual_frame_hashes)
        if actual_equivalence != manifest.decoded_frames_sha256:
            raise MediaIntegrityError("decoded frame equivalence hash mismatch")
        if (
            normalized.input_sha256 != manifest.input_sha256
            or normalized.processing_config_sha256 != manifest.processing_config_sha256
            or normalized.tool_versions != manifest.tool_versions
            or normalized.equivalence_rule != manifest.equivalence_rule
        ):
            raise MediaIntegrityError("normalized media and manifest evidence mismatch")
        return MediaProcessingResult(normalized_media=normalized, frame_manifest=manifest)
    except MediaIntegrityError:
        raise
    except (OSError, ValidationError, ValueError) as exc:
        raise MediaIntegrityError(f"invalid media publication: {exc}") from exc


def verify_normalized_equivalence(output_dir: Path, runner: MediaRunner) -> NormalizedMedia:
    """Re-decode every frame with the recorded toolchain and verify equivalence evidence."""

    output_dir = Path(output_dir)
    normalized = verify_published_media(output_dir).normalized_media
    if runner.tool_versions() != normalized.tool_versions:
        raise MediaIntegrityError("verification toolchain does not match recorded versions")
    actual = frame_sequence_sha256(runner.frame_hashes(output_dir / normalized.media_path))
    if actual != normalized.decoded_frames_sha256:
        raise MediaIntegrityError("normalized decoded frame equivalence hash mismatch")
    return normalized
