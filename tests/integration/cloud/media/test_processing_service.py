"""Atomic, deterministic media processing with a controlled synthetic toolchain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class FakeRunner:
    normalized_bytes: bytes = b"synthetic normalized media v1"
    fail_decode: bool = False
    visible_target: Path | None = None

    def tool_versions(self):
        from billiards_ai.shared.models.media import ToolVersions

        return ToolVersions(
            ffmpeg="7.1.1",
            ffprobe="7.1.1",
            decoder="ffmpeg-h264-7.1.1",
        )

    def probe(self, _source):
        from billiards_ai.cloud.media import MediaProbe
        from billiards_ai.shared.models.media import MediaResolution, TimeBase

        if Path(_source).name == "normalized.mp4":
            return MediaProbe(
                container="mp4",
                video_codec="h264",
                video_profile="High",
                pixel_format="yuv420p",
                resolution=MediaResolution(width_px=1280, height_px=720),
                orientation_degrees=0,
                time_base=TimeBase(numerator=1, denominator=1_000),
                frame_pts=(0, 33, 100, 200),
                audio_present=False,
            )
        return MediaProbe(
            container="mov,mp4,m4a,3gp,3g2,mj2",
            video_codec="h264",
            video_profile="High",
            pixel_format="yuv420p",
            resolution=MediaResolution(width_px=1920, height_px=1080),
            orientation_degrees=90,
            time_base=TimeBase(numerator=1, denominator=90_000),
            frame_pts=(90_000, 93_000, 99_000, 108_000),
            audio_present=True,
        )

    def normalize(self, _source, destination, _config, _video_stream_index=0):
        if self.visible_target is not None:
            assert not self.visible_target.exists()
        destination.write_bytes(self.normalized_bytes)

    def frame_hashes(self, _source):
        return ("a" * 64, "b" * 64, "c" * 64, "d" * 64)

    def extract_sample_frames(self, _source, destination, decode_indices):
        from billiards_ai.cloud.media import ToolExecutionError

        if self.visible_target is not None:
            assert not self.visible_target.exists()
        if self.fail_decode:
            raise ToolExecutionError("synthetic decoder failure")
        destination.mkdir()
        paths = []
        for index, decode_index in enumerate(decode_indices):
            payload = (b"A", b"B", b"C", b"D")[decode_index]
            path = destination / f"{index:08d}.png"
            path.write_bytes(b"synthetic-png-" + payload)
            paths.append(path)
        return tuple(paths)


def _process(tmp_path: Path, runner: FakeRunner, name: str):
    from billiards_ai.cloud.media import (
        MediaProcessingConfig,
        MediaProcessingLimits,
        MediaProcessingService,
    )

    source = tmp_path / "synthetic-input.mp4"
    if not source.exists():
        source.write_bytes(b"synthetic 1080p VFR input without people")
    output = tmp_path / name
    runner.visible_target = output
    result = MediaProcessingService(
        runner,
        limits=MediaProcessingLimits(
            max_input_bytes=10_000_000,
            max_decoded_frames=10_000,
            max_sampled_frames=10_000,
        ),
    ).process(
        input_path=source,
        output_dir=output,
        config=MediaProcessingConfig(interval_ms=50),
    )
    return source, output, result


def test_processing_publishes_complete_versioned_contract_atomically(tmp_path):
    from billiards_ai.shared.models.media import FrameRateStrategy

    _, output, result = _process(tmp_path, FakeRunner(), "published")

    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*")) == [
        "frame-manifest.json",
        "frames",
        "frames/000000.png",
        "frames/000001.png",
        "frames/000002.png",
        "frames/000003.png",
        "frames/000004.png",
        "normalized-media.json",
        "normalized.mp4",
    ]
    assert result.normalized_media.audio_removed is True
    assert result.normalized_media.source_orientation_degrees == 90
    assert result.normalized_media.orientation_degrees == 0
    assert result.normalized_media.time_base.denominator == 1_000
    assert result.frame_manifest.source_time_base.denominator == 90_000
    assert (
        result.normalized_media.frame_rate_strategy is FrameRateStrategy.PRESERVE_VFR_SAMPLE_BY_PTS
    )
    assert [item.source_pts for item in result.frame_manifest.frames] == [
        90_000,
        99_000,
        99_000,
        108_000,
        108_000,
    ]
    assert (
        result.normalized_media.decoded_frames_sha256 != result.frame_manifest.decoded_frames_sha256
    )
    assert not any("staging" in path.name or path.name == "decoded" for path in tmp_path.rglob("*"))


def test_repeated_runs_have_identical_manifest_and_decoded_equivalence(tmp_path):
    _, _, first = _process(tmp_path, FakeRunner(normalized_bytes=b"encoding-a"), "run-a")
    _, _, second = _process(tmp_path, FakeRunner(normalized_bytes=b"encoding-b"), "run-b")

    assert first.frame_manifest == second.frame_manifest
    assert first.normalized_media.content_sha256 != second.normalized_media.content_sha256
    assert first.normalized_media.is_equivalent_to(second.normalized_media)


def test_decode_failure_is_logged_and_leaves_no_consumable_output(tmp_path, caplog):
    from billiards_ai.cloud.media import ToolExecutionError

    target = tmp_path / "failed"
    with caplog.at_level("ERROR"), pytest.raises(ToolExecutionError, match="decoder failure"):
        _process(tmp_path, FakeRunner(fail_decode=True), target.name)

    assert not target.exists()
    assert "decoder failure" in caplog.text
    assert not tuple(tmp_path.glob(".*.staging-*"))


def test_corrupt_and_unsupported_inputs_are_typed_and_never_published(tmp_path):
    from billiards_ai.cloud.media import (
        CorruptMediaError,
        MediaProcessingConfig,
        MediaProcessingLimits,
        MediaProcessingService,
        UnsupportedMediaError,
    )

    class RejectedRunner(FakeRunner):
        def __init__(self, error):
            super().__init__()
            self.error = error

        def probe(self, _source):
            raise self.error

    source = tmp_path / "bad.bin"
    source.write_bytes(b"not real media")
    for error, name in (
        (CorruptMediaError("truncated synthetic container"), "corrupt"),
        (UnsupportedMediaError("unsupported synthetic codec"), "unsupported"),
    ):
        output = tmp_path / name
        with pytest.raises(type(error), match="synthetic"):
            MediaProcessingService(
                RejectedRunner(error),
                limits=MediaProcessingLimits(
                    max_input_bytes=10_000_000,
                    max_decoded_frames=10_000,
                    max_sampled_frames=10_000,
                ),
            ).process(
                input_path=source,
                output_dir=output,
                config=MediaProcessingConfig(interval_ms=100),
            )
        assert not output.exists()


def test_published_hash_tampering_is_detected(tmp_path):
    from billiards_ai.cloud.media import MediaIntegrityError, verify_published_media

    _, output, _ = _process(tmp_path, FakeRunner(), "published")
    (output / "frames" / "000002.png").write_bytes(b"tampered")

    with pytest.raises(MediaIntegrityError, match="hash"):
        verify_published_media(output)


def test_full_decoded_equivalence_is_independently_verifiable(tmp_path):
    from billiards_ai.cloud.media import (
        MediaIntegrityError,
        verify_normalized_equivalence,
    )

    _, output, result = _process(tmp_path, FakeRunner(), "published")

    assert verify_normalized_equivalence(output, FakeRunner()) == result.normalized_media

    class ChangedDecodeRunner(FakeRunner):
        def frame_hashes(self, _source):
            return ("a" * 64, "b" * 64, "f" * 64, "d" * 64)

    with pytest.raises(MediaIntegrityError, match="decoded frame equivalence"):
        verify_normalized_equivalence(output, ChangedDecodeRunner())


def test_processing_never_materializes_all_decoded_frames_as_pngs(tmp_path):
    class CapturingRunner(FakeRunner):
        decoded_indices: tuple[int, ...] = ()

        def extract_sample_frames(self, source, destination, decode_indices):
            self.decoded_indices = tuple(decode_indices)
            return super().extract_sample_frames(source, destination, decode_indices)

    runner = CapturingRunner()
    _, output, _ = _process(tmp_path, runner, "published")

    assert runner.decoded_indices == (0, 2, 3)
    assert not (output / "decoded").exists()
    assert len(tuple((output / "frames").glob("*.png"))) == 5


def test_injected_limits_reject_oversized_input_before_probe(tmp_path):
    from billiards_ai.cloud.media import (
        MediaProcessingConfig,
        MediaProcessingLimits,
        MediaProcessingService,
        ResourceLimitError,
    )

    source = tmp_path / "large.mp4"
    source.write_bytes(b"0123456789")
    with pytest.raises(ResourceLimitError, match="input size"):
        MediaProcessingService(
            FakeRunner(),
            limits=MediaProcessingLimits(
                max_input_bytes=9,
                max_decoded_frames=10,
                max_sampled_frames=10,
            ),
        ).process(
            input_path=source,
            output_dir=tmp_path / "published",
            config=MediaProcessingConfig(interval_ms=100),
        )
