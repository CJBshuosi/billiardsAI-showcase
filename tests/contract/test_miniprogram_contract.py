import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from billiards_ai.shared import models
from billiards_ai.shared.models import EdgeSyncEnvelopeV1, canonical_sync_hash

FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "synthetic" / "miniprogram" / "session-facts-v1.json"
)


def test_edge_sync_contract_rejects_private_identity_and_unknown_fields() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["openid"] = "must-not-cross-boundary"

    with pytest.raises(ValidationError, match="openid"):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_edge_sync_hash_is_canonical_after_round_trip() -> None:
    first = EdgeSyncEnvelopeV1.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    reordered = {
        "media": first.model_dump(mode="json")["media"],
        "session": first.model_dump(mode="json")["session"],
        "schema_version": "1.0",
    }
    second = EdgeSyncEnvelopeV1.model_validate(reordered)

    assert canonical_sync_hash(first) == canonical_sync_hash(second)
    assert len(canonical_sync_hash(first)) == 64


def test_edge_sync_contract_preserves_session_shot_and_clip_lineage() -> None:
    envelope = EdgeSyncEnvelopeV1.model_validate_json(FIXTURE.read_text(encoding="utf-8"))

    assert envelope.session.session_id == "session-0001"
    assert envelope.session.shots[0].shot_id == "shot-0001"
    assert envelope.session.shots[0].live_job_id == "job-development-0001"
    assert envelope.session.postgame_manifest_sha256 == "b" * 64
    assert envelope.session.evidence_source_kind == "development_replay"
    assert envelope.media[0].session_id == envelope.session.session_id
    assert envelope.media[0].shot_id == envelope.session.shots[0].shot_id


def test_edge_sync_contract_accepts_only_canonical_media_sources() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["media"][0]["source_kind"] = "development_replay"
    assert EdgeSyncEnvelopeV1.model_validate(payload).media[0].source_kind == "development_replay"

    payload["media"][0]["source_kind"] = "usb_live"
    payload["session"]["evidence_source_kind"] = "usb_live"
    assert EdgeSyncEnvelopeV1.model_validate(payload).media[0].source_kind == "usb_live"

    payload["media"][0]["source_kind"] = "development_presentation"
    with pytest.raises(ValidationError, match="source_kind"):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_edge_sync_contract_rejects_missing_or_inconsistent_explicit_evidence_source() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["session"]["evidence_source_kind"]
    with pytest.raises(ValidationError, match="evidence_source_kind"):
        EdgeSyncEnvelopeV1.model_validate(payload)

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["session"]["evidence_source_kind"] = "usb_live"
    with pytest.raises(ValidationError, match="evidence source"):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_edge_sync_contract_accepts_legacy_replay_media_without_saved_lineage() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["session"]["postgame_manifest_sha256"]
    del payload["session"]["shots"][0]["live_job_id"]

    envelope = EdgeSyncEnvelopeV1.model_validate(payload)

    assert envelope.session.evidence_source_kind == "development_replay"
    assert envelope.session.postgame_manifest_sha256 is None
    assert envelope.session.shots[0].live_job_id is None


def test_edge_sync_contract_requires_saved_lineage_for_nonempty_facts_only_usb() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["media"] = []
    payload["session"]["evidence_source_kind"] = "usb_live"
    payload["session"]["limitations"] = [
        item for item in payload["session"]["limitations"] if item != "development_replay_media"
    ]
    del payload["session"]["shots"][0]["live_job_id"]

    with pytest.raises(ValidationError, match="live job"):
        EdgeSyncEnvelopeV1.model_validate(payload)

    payload["session"]["shots"][0]["live_job_id"] = "job-development-0001"
    del payload["session"]["postgame_manifest_sha256"]
    with pytest.raises(ValidationError, match="postgame"):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_edge_sync_contract_rejects_media_longer_than_five_minutes() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["media"][0].update({"end_ms": 300_001, "duration_ms": 300_001})

    with pytest.raises(ValidationError, match="300000"):
        EdgeSyncEnvelopeV1.model_validate(payload)


@pytest.mark.parametrize(
    ("status", "counts", "report_available"),
    [
        ("PROCESSING", (None, None, None), False),
        ("AVAILABLE", (3, 3, 0), True),
        ("PARTIAL", (3, 2, 1), True),
        ("FAILED_RETRYING", (3, 0, 2), False),
    ],
)
def test_session_analysis_projection_accepts_each_public_state(
    status: str,
    counts: tuple[int | None, int | None, int | None],
    report_available: bool,
) -> None:
    assert hasattr(models, "SessionAnalysisProjectionV1")
    projection_type = models.SessionAnalysisProjectionV1
    total, available, retrying = counts
    projection = projection_type.model_validate(
        {
            "schema_version": "1.0",
            "session_id": "session-0001",
            "revision": 2,
            "status": status,
            "total_shots": total,
            "available_shots": available,
            "retrying_shots": retrying,
            "report_available": report_available,
            "deadline_at": "2026-09-06T12:02:00Z",
            "updated_at": "2026-09-06T12:00:30Z",
        }
    )

    assert projection.status == status


def test_session_analysis_projection_rejects_partial_or_impossible_counts() -> None:
    assert hasattr(models, "SessionAnalysisProjectionV1")
    projection_type = models.SessionAnalysisProjectionV1
    base = {
        "schema_version": "1.0",
        "session_id": "session-0001",
        "revision": 2,
        "status": "PROCESSING",
        "total_shots": None,
        "available_shots": None,
        "retrying_shots": None,
        "report_available": False,
        "deadline_at": "2026-09-06T12:02:00Z",
        "updated_at": "2026-09-06T12:00:30Z",
    }
    with pytest.raises(ValidationError, match="counts"):
        projection_type.model_validate({**base, "total_shots": 3})
    with pytest.raises(ValidationError, match="shot count"):
        projection_type.model_validate(
            {
                **base,
                "status": "PARTIAL",
                "total_shots": 2,
                "available_shots": 2,
                "retrying_shots": 1,
                "report_available": True,
            }
        )


@pytest.mark.parametrize(
    ("status", "available", "retrying", "report_available"),
    [
        ("PROCESSING", 1, 0, False),
        ("PROCESSING", 0, 1, False),
        ("PROCESSING", 0, 0, True),
        ("PARTIAL", 0, 1, True),
        ("PARTIAL", 3, 0, True),
        ("PARTIAL", 1, 1, False),
        ("FAILED_RETRYING", 1, 1, False),
        ("FAILED_RETRYING", 0, 0, False),
        ("FAILED_RETRYING", 0, 1, True),
    ],
)
def test_session_analysis_projection_rejects_state_semantic_conflicts(
    status: str,
    available: int,
    retrying: int,
    report_available: bool,
) -> None:
    projection_type = models.SessionAnalysisProjectionV1
    with pytest.raises(ValidationError, match="analysis status"):
        projection_type.model_validate(
            {
                "schema_version": "1.0",
                "session_id": "session-0001",
                "revision": 2,
                "status": status,
                "total_shots": 3,
                "available_shots": available,
                "retrying_shots": retrying,
                "report_available": report_available,
                "deadline_at": "2026-09-06T12:02:00Z",
                "updated_at": "2026-09-06T12:00:30Z",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "../secret.mp4"),
        ("relative_path", "/absolute.mp4"),
        ("relative_path", "clips\\shot.mp4"),
        ("content_sha256", "A" * 64),
        ("byte_size", True),
        ("duration_ms", 0),
    ],
)
def test_edge_sync_contract_rejects_unsafe_media_fields(field: str, value: object) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["media"][0][field] = value

    with pytest.raises(ValidationError):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_edge_sync_contract_rejects_media_outside_session_lineage() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["media"][0]["shot_id"] = "shot-does-not-exist"

    with pytest.raises(ValidationError, match="media shot must belong"):
        EdgeSyncEnvelopeV1.model_validate(payload)


def test_dynamic_table_code_generation_exists_only_in_one_time_provisioning() -> None:
    root = Path("apps/miniprogram")
    allowed = root / "provisioning" / "table-code" / "index.js"
    prohibited = ("getUnlimited", "/join-token", "session_join_tokens", "qr_image_base64")

    assert "getUnlimited" in allowed.read_text(encoding="utf-8")
    for path in root.rglob("*.js"):
        if path == allowed or "node_modules" in path.parts or "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in source for token in prohibited), path
