"""Executable AD-13h contracts for normalized media and frame manifests."""

import hashlib

import pytest
from pydantic import ValidationError


def _api():
    from billiards_ai.shared.models.media import (
        FrameManifest,
        FrameRateStrategy,
        FrameRecord,
        MediaEquivalenceRule,
        MediaResolution,
        NormalizedMedia,
        TimeBase,
        ToolVersions,
    )

    return {
        "FrameManifest": FrameManifest,
        "FrameRateStrategy": FrameRateStrategy,
        "FrameRecord": FrameRecord,
        "MediaEquivalenceRule": MediaEquivalenceRule,
        "MediaResolution": MediaResolution,
        "NormalizedMedia": NormalizedMedia,
        "TimeBase": TimeBase,
        "ToolVersions": ToolVersions,
    }


def _values(api: dict):
    frame_hashes = ("1" * 64, "2" * 64)
    equivalence_hash = hashlib.sha256(("\n".join(frame_hashes) + "\n").encode()).hexdigest()
    tools = api["ToolVersions"](
        ffmpeg="7.1.1",
        ffprobe="7.1.1",
        decoder="ffmpeg-h264-7.1.1",
    )
    frames = (
        api["FrameRecord"](
            index=0,
            requested_timestamp_ms=0,
            source_pts=90_000,
            image_path="frames/000000.png",
            content_sha256=frame_hashes[0],
        ),
        api["FrameRecord"](
            index=1,
            requested_timestamp_ms=100,
            source_pts=99_000,
            image_path="frames/000001.png",
            content_sha256=frame_hashes[1],
        ),
    )
    common = {
        "input_sha256": "a" * 64,
        "processing_config_sha256": "b" * 64,
        "tool_versions": tools,
        "equivalence_rule": api["MediaEquivalenceRule"].DECODED_FRAME_SHA256_V1,
        "decoded_frames_sha256": equivalence_hash,
    }
    manifest = api["FrameManifest"](
        **common,
        interval_ms=100,
        source_time_base=api["TimeBase"](numerator=1, denominator=90_000),
        first_source_pts=90_000,
        frames=frames,
    )
    normalized = api["NormalizedMedia"](
        **common,
        media_path="normalized.mp4",
        container="mp4",
        video_codec="h264",
        video_profile="High",
        pixel_format="yuv420p",
        resolution=api["MediaResolution"](width_px=1280, height_px=720),
        source_orientation_degrees=90,
        orientation_degrees=0,
        time_base=api["TimeBase"](numerator=1, denominator=90_000),
        frame_rate_strategy=api["FrameRateStrategy"].PRESERVE_VFR_SAMPLE_BY_PTS,
        audio_removed=True,
        content_sha256="c" * 64,
    )
    return normalized, manifest


def test_media_contract_is_versioned_strict_frozen_and_complete():
    api = _api()
    normalized, manifest = _values(api)

    assert normalized.schema_version == "1.0"
    assert normalized.audio_removed is True
    assert normalized.orientation_degrees == 0
    assert manifest.schema_version == "1.0"
    assert manifest.frames[1].source_pts == 99_000

    with pytest.raises(ValidationError):
        api["NormalizedMedia"].model_validate(normalized.model_dump() | {"camera_vendor": "x"})
    with pytest.raises(ValidationError):
        manifest.frames[0].source_pts = 0
    with pytest.raises(ValidationError):
        api["TimeBase"](numerator="1", denominator=90_000)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/tmp/frame.png", "../frame.png", "frames/../x.png", "..\\secret.png", "C:\\secret.png"],
)
def test_contract_rejects_paths_outside_the_atomic_publication(unsafe_path):
    api = _api()
    _, manifest = _values(api)

    with pytest.raises(ValidationError, match="controlled relative path"):
        api["FrameRecord"].model_validate(
            manifest.frames[0].model_dump() | {"image_path": unsafe_path}
        )


def test_manifest_rejects_non_contiguous_indices_and_pts_before_requested_target():
    api = _api()
    _, manifest = _values(api)
    values = manifest.model_dump()
    bad_index = dict(values["frames"][1]) | {"index": 3}

    with pytest.raises(ValidationError, match="contiguous"):
        api["FrameManifest"].model_validate(values | {"frames": (values["frames"][0], bad_index)})

    too_early = dict(values["frames"][1]) | {"source_pts": 98_999}
    with pytest.raises(ValidationError, match="before requested timestamp"):
        api["FrameManifest"].model_validate(values | {"frames": (values["frames"][0], too_early)})


def test_decoded_frame_equivalence_can_hold_when_encoded_bytes_differ():
    api = _api()
    first, _ = _values(api)
    second = first.model_copy(update={"content_sha256": "d" * 64})

    assert first.content_sha256 != second.content_sha256
    assert first.is_equivalent_to(second)

    changed_evidence = second.model_copy(update={"decoded_frames_sha256": "e" * 64})
    assert not first.is_equivalent_to(changed_evidence)
