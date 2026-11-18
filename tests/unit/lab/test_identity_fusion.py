from billiards_ai.lab.identity_fusion import (
    FusionStatus,
    IdentityFrameEvidence,
    classify_and_fuse_track_identities,
    fuse_track_identities,
)


class _IdentityAdapter:
    def resolve_identities(self, observations: tuple[dict[str, object], ...]) -> tuple[dict, ...]:
        return tuple(
            {
                "track_id": item["track_id"],
                "exact_identity": "8",
                "group": "eight",
                "confidence": 0.95,
            }
            for item in observations
        )


def _evidence(
    frame_id: int,
    track_id: str,
    identity: str | None,
    group: str,
    confidence: float,
) -> IdentityFrameEvidence:
    return IdentityFrameEvidence(
        frame_id=frame_id,
        track_id=track_id,
        exact_identity=identity,
        group=group,
        confidence=confidence,
        source_ref=f"frame:{frame_id}",
    )


def test_three_consistent_frames_resolve_track_group_and_exact_vote() -> None:
    result = fuse_track_identities(
        [
            _evidence(1, "t1", None, "stripe", 0.84),
            _evidence(2, "t1", None, "stripe", 0.91),
            _evidence(3, "t1", "10", "stripe", 0.88),
        ]
    )["t1"]
    assert result.identity == "10"
    assert result.group == "stripe"
    assert result.status is FusionStatus.RESOLVED
    assert result.supporting_frame_ids == (1, 2, 3)


def test_equal_conflicting_exact_votes_become_ambiguous() -> None:
    result = fuse_track_identities(
        [
            _evidence(1, "t1", "2", "solid", 0.90),
            _evidence(2, "t1", "10", "stripe", 0.90),
        ]
    )["t1"]
    assert result.status is FusionStatus.AMBIGUOUS
    assert result.identity is None
    assert result.group == "unknown"


def test_group_only_evidence_is_useful_without_forcing_number() -> None:
    result = fuse_track_identities(
        [
            _evidence(1, "t1", None, "solid", 0.8),
            _evidence(2, "t1", None, "solid", 0.9),
        ]
    )["t1"]
    assert result.status is FusionStatus.GROUP_ONLY
    assert result.identity is None
    assert result.group == "solid"


def test_output_order_and_frame_evidence_are_deterministic() -> None:
    result = fuse_track_identities(
        [
            _evidence(4, "z", None, "stripe", 0.8),
            _evidence(2, "a", "1", "solid", 0.9),
            _evidence(1, "a", None, "solid", 0.8),
        ]
    )
    assert tuple(result) == ("a", "z")
    assert result["a"].supporting_frame_ids == (1, 2)


def test_frame_crops_flow_through_injected_classifier_then_temporal_fusion() -> None:
    result = classify_and_fuse_track_identities(
        adapter=_IdentityAdapter(),
        observations=(
            {"frame_id": 1, "track_id": "track-8", "crop": b"a", "source_ref": "v:1"},
            {"frame_id": 2, "track_id": "track-8", "crop": b"b", "source_ref": "v:2"},
        ),
    )

    assert result["track-8"].identity == "8"
    assert result["track-8"].status is FusionStatus.RESOLVED
    assert result["track-8"].supporting_frame_ids == (1, 2)
