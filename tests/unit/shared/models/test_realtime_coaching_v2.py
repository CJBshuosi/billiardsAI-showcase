from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from billiards_ai.shared.models.overlay import (
    OverlayCoordinateSpace,
    OverlayLineStyle,
    OverlayPayloadV1,
    OverlayPrimitiveKind,
    OverlayPrimitiveV1,
    OverlaySemanticRole,
    TablePointMmV1,
    TableZoneMmV1,
)
from billiards_ai.shared.models.realtime import (
    CoachingDeliveryStatus,
    CoachingEventType,
    CoachingOutcomeStatus,
    RealtimeCoachingEvent,
    RealtimeCoachingEventV2,
    RealtimeCoachingLineageV1,
    RealtimeFeedbackKind,
    SpeechCueV1,
    canonical_realtime_coaching_event_bytes,
    decode_realtime_coaching_event,
    issue_realtime_coaching_event_v2,
)

SESSION_ID = UUID("00000000-0000-4000-8000-000000000002")
PLAN_ID = UUID("00000000-0000-4000-8000-000000000004")
BLOCK_ID = UUID("00000000-0000-4000-8000-000000000006")
SHOT_ID = UUID("00000000-0000-4000-8000-000000000008")
BUNDLE_ID = UUID("00000000-0000-4000-8000-000000000010")


def _lineage() -> RealtimeCoachingLineageV1:
    return RealtimeCoachingLineageV1(
        admission_id="admission-1",
        bundle_id=BUNDLE_ID,
        bundle_content_sha256="a" * 64,
        projection_sha256="b" * 64,
        session_id=SESSION_ID,
        plan_id=PLAN_ID,
        plan_version=3,
        block_id=BLOCK_ID,
        block_order=1,
        drill_id="target-zone-01",
        drill_revision=2,
        drill_content_sha256="c" * 64,
        shot_id=SHOT_ID,
        shot_sequence=7,
        evidence_sha256="d" * 64,
        decision_id="e" * 64,
        effect_id="f" * 64,
    )


def _overlay() -> OverlayPayloadV1:
    return OverlayPayloadV1(
        coordinate_space=OverlayCoordinateSpace.TABLE_MM,
        outcome_status=CoachingOutcomeStatus.SUCCESS,
        delivery_status=CoachingDeliveryStatus.NORMAL,
        primitives=(
            OverlayPrimitiveV1(
                kind=OverlayPrimitiveKind.POINT,
                semantic_role=OverlaySemanticRole.ACTUAL_CUE_BALL_END,
                label_code="actual-cue-ball-end",
                line_style=OverlayLineStyle.SOLID,
                point=TablePointMmV1(x_mm=150, y_mm=350),
            ),
            OverlayPrimitiveV1(
                kind=OverlayPrimitiveKind.ZONE,
                semantic_role=OverlaySemanticRole.TARGET_ZONE,
                label_code="published-target-zone",
                line_style=OverlayLineStyle.DASHED,
                zone=TableZoneMmV1(
                    minimum_x_mm=100,
                    maximum_x_mm=200,
                    minimum_y_mm=300,
                    maximum_y_mm=400,
                ),
            ),
        ),
    )


def _speech() -> SpeechCueV1:
    return SpeechCueV1(
        locale="zh-CN",
        cue_code="shot-success",
        outcome_status=CoachingOutcomeStatus.SUCCESS,
        delivery_status=CoachingDeliveryStatus.NORMAL,
        text="本杆达到已发布训练标准。",
    )


def _event() -> RealtimeCoachingEventV2:
    return issue_realtime_coaching_event_v2(
        sequence=9,
        event_type=CoachingEventType.POST_SHOT_FEEDBACK,
        outcome_status=CoachingOutcomeStatus.SUCCESS,
        delivery_status=CoachingDeliveryStatus.NORMAL,
        lineage=_lineage(),
        overlay=_overlay(),
        speech=_speech(),
        text_summary="本杆达到已发布训练标准。",
        created_monotonic_ms=7500,
    )


def test_v1_contract_remains_unchanged_and_decodes_explicitly() -> None:
    event = RealtimeCoachingEvent(
        event_id=UUID("00000000-0000-4000-8000-000000000012"),
        session_id=SESSION_ID,
        shot_sequence=7,
        kind=RealtimeFeedbackKind.TEXT,
        sequence=9,
        payload_code="legacy-text",
        created_at=datetime(2026, 8, 10, tzinfo=UTC),
    )

    decoded = decode_realtime_coaching_event(canonical_realtime_coaching_event_bytes(event))

    assert type(decoded) is RealtimeCoachingEvent
    assert tuple(type(decoded).model_fields) == (
        "schema_version",
        "event_id",
        "session_id",
        "shot_sequence",
        "kind",
        "sequence",
        "payload_code",
        "created_at",
    )


def test_overlay_is_table_mm_only_strict_frozen_and_non_color_dependent() -> None:
    overlay = _overlay()

    assert overlay.coordinate_space is OverlayCoordinateSpace.TABLE_MM
    assert tuple(item.semantic_role for item in overlay.primitives) == (
        OverlaySemanticRole.ACTUAL_CUE_BALL_END,
        OverlaySemanticRole.TARGET_ZONE,
    )
    assert tuple(item.line_style for item in overlay.primitives) == (
        OverlayLineStyle.SOLID,
        OverlayLineStyle.DASHED,
    )
    with pytest.raises(ValidationError):
        TablePointMmV1(x_mm=150.0, y_mm=350)
    with pytest.raises(ValidationError):
        TablePointMmV1(x_mm=150, y_mm=350, pixel_x=12)
    with pytest.raises(ValidationError):
        overlay.outcome_status = CoachingOutcomeStatus.FAILURE


@pytest.mark.parametrize(
    "zone",
    [
        {"minimum_x_mm": 201, "maximum_x_mm": 200, "minimum_y_mm": 300, "maximum_y_mm": 400},
        {"minimum_x_mm": -1, "maximum_x_mm": 200, "minimum_y_mm": 300, "maximum_y_mm": 400},
        {"minimum_x_mm": 100, "maximum_x_mm": 2541, "minimum_y_mm": 300, "maximum_y_mm": 400},
    ],
)
def test_overlay_zone_rejects_invalid_or_off_table_geometry(zone: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        TableZoneMmV1(**zone)


def test_overlay_primitive_requires_exactly_the_geometry_for_its_kind() -> None:
    point = TablePointMmV1(x_mm=150, y_mm=350)
    zone = TableZoneMmV1(
        minimum_x_mm=100,
        maximum_x_mm=200,
        minimum_y_mm=300,
        maximum_y_mm=400,
    )
    common = {
        "semantic_role": OverlaySemanticRole.ACTUAL_CUE_BALL_END,
        "label_code": "actual-cue-ball-end",
        "line_style": OverlayLineStyle.SOLID,
    }

    with pytest.raises(ValidationError):
        OverlayPrimitiveV1(kind=OverlayPrimitiveKind.POINT, point=point, zone=zone, **common)
    with pytest.raises(ValidationError):
        OverlayPrimitiveV1(kind=OverlayPrimitiveKind.ZONE, **common)
    with pytest.raises(ValidationError, match="semantic role"):
        OverlayPrimitiveV1(
            kind=OverlayPrimitiveKind.ZONE,
            zone=zone,
            **common,
        )
    with pytest.raises(ValidationError, match="STATUS"):
        OverlayPrimitiveV1(
            kind=OverlayPrimitiveKind.POINT,
            semantic_role=OverlaySemanticRole.STATUS,
            label_code="status",
            line_style=OverlayLineStyle.DOTTED,
            point=point,
        )


def test_overlay_roles_must_be_unique_and_canonically_ordered() -> None:
    actual, target = _overlay().primitives
    with pytest.raises(ValidationError):
        OverlayPayloadV1(
            coordinate_space=OverlayCoordinateSpace.TABLE_MM,
            outcome_status=CoachingOutcomeStatus.SUCCESS,
            delivery_status=CoachingDeliveryStatus.NORMAL,
            primitives=(target, actual),
        )
    with pytest.raises(ValidationError):
        OverlayPayloadV1(
            coordinate_space=OverlayCoordinateSpace.TABLE_MM,
            outcome_status=CoachingOutcomeStatus.SUCCESS,
            delivery_status=CoachingDeliveryStatus.NORMAL,
            primitives=(actual, actual),
        )


def test_v2_identity_content_hash_and_canonical_round_trip_are_stable() -> None:
    first = _event()
    second = _event()
    first_bytes = canonical_realtime_coaching_event_bytes(first)

    assert first.event_id.version == 4
    assert first == second
    assert first_bytes == canonical_realtime_coaching_event_bytes(second)
    assert decode_realtime_coaching_event(first_bytes) == first
    assert len(first.idempotency_key) == 64
    assert len(first.content_sha256) == 64
    assert b"raw_observation" not in first_bytes
    assert b"trajectory" not in first_bytes
    assert b"pixel" not in first_bytes
    for forbidden in (b"media", b"locator", b"url", b"prompt", b"token"):
        assert forbidden not in first_bytes

    def property_names(node: object) -> set[str]:
        if isinstance(node, dict):
            names = set(node.get("properties", {}))
            return names | set().union(*(property_names(value) for value in node.values()))
        if isinstance(node, list):
            return set().union(*(property_names(value) for value in node))
        return set()

    schema_fields = {name.lower() for name in property_names(_event().model_json_schema())}
    assert schema_fields.isdisjoint(
        {"raw_observation", "trajectory", "pixel", "media", "locator", "url", "prompt", "token"}
    )


def test_v2_rejects_status_mismatch_and_incomplete_post_shot_lineage() -> None:
    bad_speech = _speech().model_copy(update={"outcome_status": CoachingOutcomeStatus.FAILURE})
    with pytest.raises(ValueError, match="status"):
        issue_realtime_coaching_event_v2(
            sequence=9,
            event_type=CoachingEventType.POST_SHOT_FEEDBACK,
            outcome_status=CoachingOutcomeStatus.SUCCESS,
            delivery_status=CoachingDeliveryStatus.NORMAL,
            lineage=_lineage(),
            overlay=_overlay(),
            speech=bad_speech,
            text_summary="本杆达到已发布训练标准。",
            created_monotonic_ms=7500,
        )
    with pytest.raises(ValidationError):
        RealtimeCoachingLineageV1(**(_lineage().model_dump(exclude={"evidence_sha256"})))


@pytest.mark.parametrize(
    ("event_type", "outcome", "delivery"),
    [
        (
            CoachingEventType.POST_SHOT_FEEDBACK,
            CoachingOutcomeStatus.SUCCESS,
            CoachingDeliveryStatus.ERROR,
        ),
        (
            CoachingEventType.POST_SHOT_FEEDBACK,
            CoachingOutcomeStatus.ABSTAINED,
            CoachingDeliveryStatus.NORMAL,
        ),
        (
            CoachingEventType.SETUP,
            CoachingOutcomeStatus.FAILURE,
            CoachingDeliveryStatus.NORMAL,
        ),
    ],
)
def test_v2_rejects_impossible_event_outcome_delivery_combinations(
    event_type: CoachingEventType,
    outcome: CoachingOutcomeStatus,
    delivery: CoachingDeliveryStatus,
) -> None:
    speech = SpeechCueV1(
        cue_code="invalid-combination",
        outcome_status=outcome,
        delivery_status=delivery,
        text="结构化状态测试。",
    )
    event = _event().model_copy(
        update={
            "event_type": event_type,
            "outcome_status": outcome,
            "delivery_status": delivery,
            "overlay": None,
            "speech": speech,
            "text_summary": speech.text,
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        event.statuses_lineage_and_hashes_are_consistent()


def test_v2_requires_machine_outcome_lineage_and_consistent_speech_text() -> None:
    no_decision = _lineage().model_copy(update={"decision_id": None})
    unsupported_speech = SpeechCueV1(
        cue_code="shot-unsupported-safe-pause",
        outcome_status=CoachingOutcomeStatus.UNSUPPORTED,
        delivery_status=CoachingDeliveryStatus.DEGRADED,
        text="本杆不计为失败。",
    )
    with pytest.raises(ValueError, match="machine outcome"):
        issue_realtime_coaching_event_v2(
            sequence=9,
            event_type=CoachingEventType.SYSTEM_STATUS,
            outcome_status=CoachingOutcomeStatus.UNSUPPORTED,
            delivery_status=CoachingDeliveryStatus.DEGRADED,
            lineage=no_decision,
            overlay=None,
            speech=unsupported_speech,
            text_summary="本杆不计为失败。",
            created_monotonic_ms=7500,
        )

    with pytest.raises(ValueError, match="require a speech cue"):
        issue_realtime_coaching_event_v2(
            sequence=9,
            event_type=CoachingEventType.SYSTEM_STATUS,
            outcome_status=CoachingOutcomeStatus.INFORMATIONAL,
            delivery_status=CoachingDeliveryStatus.ERROR,
            lineage=_lineage(),
            overlay=None,
            speech=None,
            text_summary="训练指导已停止。",
            created_monotonic_ms=7500,
        )

    contradictory = _speech().model_copy(update={"text": "本杆未达到标准。"})
    with pytest.raises(ValueError, match="speech text"):
        issue_realtime_coaching_event_v2(
            sequence=9,
            event_type=CoachingEventType.POST_SHOT_FEEDBACK,
            outcome_status=CoachingOutcomeStatus.SUCCESS,
            delivery_status=CoachingDeliveryStatus.NORMAL,
            lineage=_lineage(),
            overlay=_overlay(),
            speech=contradictory,
            text_summary="本杆达到已发布训练标准。",
            created_monotonic_ms=7500,
        )


def test_public_issuer_revalidates_nested_model_copy_inputs() -> None:
    invalid = _lineage().model_copy(update={"shot_id": None})
    with pytest.raises(ValidationError, match="shot lineage"):
        issue_realtime_coaching_event_v2(
            sequence=9,
            event_type=CoachingEventType.POST_SHOT_FEEDBACK,
            outcome_status=CoachingOutcomeStatus.SUCCESS,
            delivery_status=CoachingDeliveryStatus.NORMAL,
            lineage=invalid,
            overlay=_overlay(),
            speech=_speech(),
            text_summary="本杆达到已发布训练标准。",
            created_monotonic_ms=7500,
        )


def test_decoder_rejects_tampering_noncanonical_bytes_and_unknown_major() -> None:
    encoded = canonical_realtime_coaching_event_bytes(_event())
    tampered = json.loads(encoded)
    tampered["text_summary"] = "tampered"
    tampered_bytes = json.dumps(
        tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    with pytest.raises((ValidationError, ValueError), match="content|identity"):
        decode_realtime_coaching_event(tampered_bytes)

    noncanonical = json.dumps(json.loads(encoded), ensure_ascii=False).encode()
    with pytest.raises(ValueError, match="canonical"):
        decode_realtime_coaching_event(noncanonical)

    unknown = json.dumps({"schema_version": "3.0"}, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="schema version"):
        decode_realtime_coaching_event(unknown)

    with pytest.raises(ValueError, match="maximum wire size"):
        decode_realtime_coaching_event(b" " * 65_537)


def test_canonical_serializer_rejects_unvalidated_model_copy_tampering() -> None:
    tampered = _event().model_copy(update={"text_summary": "tampered"})

    with pytest.raises(ValueError, match="speech text|content"):
        canonical_realtime_coaching_event_bytes(tampered)
