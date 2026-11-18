import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.access_identity import FileReviewerIdentityProvider
from billiards_ai.cloud.privacy.access_repository import (
    MAX_ACCESS_STORE_BYTES,
    AccessRepositoryError,
    JsonAccessRepository,
)
from billiards_ai.shared.models import (
    AccessAction,
    AccessActorRole,
    AccessAuditEvent,
    AccessDecisionResult,
    AccessReasonCode,
    MediaAssetKind,
    SessionAccessGrant,
    ThirdPartyProcessingApproval,
)
from billiards_ai.shared.models.privacy import MaterialUsePurpose

NOW = datetime(2026, 7, 28, 16, 30, tzinfo=UTC)


def _write_reviewer_credential(home: Path, *, mode: int = 0o600) -> Path:
    directory = home / ".config" / "billiards-ai"
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / "reviewer.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "principal_id": "reviewer-1",
                "roles": ["ACCESS_ADMIN", "REVIEWER"],
                "credential_id": "phase0-reviewer",
            }
        ),
        encoding="utf-8",
    )
    path.chmod(mode)
    return path


def _grant(session_id=None) -> SessionAccessGrant:
    return SessionAccessGrant(
        grant_id=uuid4(),
        reviewer_id="reviewer-1",
        session_id=session_id or uuid4(),
        reason="review session",
        granted_by="reviewer-1",
        granted_at=NOW,
    )


def _event(session_id, sequence=1) -> AccessAuditEvent:
    return AccessAuditEvent(
        event_id=uuid4(),
        sequence=sequence,
        actor_id="reviewer-1",
        actor_role=AccessActorRole.REVIEWER,
        occurred_at=NOW,
        session_id=session_id,
        action=AccessAction.CONTROLLED_REVIEW_VIEW,
        reason="manual review",
        result=AccessDecisionResult.ALLOW,
        reason_code=AccessReasonCode.AUTHORIZED,
        security_event=False,
    )


def test_reviewer_identity_comes_from_fixed_private_account_file(tmp_path: Path) -> None:
    _write_reviewer_credential(tmp_path)

    identity = FileReviewerIdentityProvider().current_identity()

    assert identity.principal_id == "reviewer-1"
    assert identity.roles == (AccessActorRole.ACCESS_ADMIN, AccessActorRole.REVIEWER)


def test_reviewer_identity_rejects_unsafe_permissions(tmp_path: Path) -> None:
    _write_reviewer_credential(tmp_path, mode=0o644)

    with pytest.raises(AccessRepositoryError):
        FileReviewerIdentityProvider().current_identity()


def test_repository_creates_revokes_and_queries_exact_session_grant(tmp_path: Path) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    grant = _grant()

    repository.create_grant(grant)
    assert repository.find_active_grant("reviewer-1", grant.session_id) == grant

    revoked = repository.revoke_grant(
        grant.grant_id,
        revoked_at=NOW,
        revocation_reason="review complete",
    )
    assert revoked.revoked_at == NOW
    assert repository.find_active_grant("reviewer-1", grant.session_id) is None


def test_repository_appends_events_with_repository_owned_sequence(tmp_path: Path) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    session_id = uuid4()

    first = repository.append_event(_event(session_id))
    second = repository.append_event(_event(session_id, sequence=999))

    assert [first.sequence, second.sequence] == [1, 2]
    assert repository.list_events(session_id=session_id) == (first, second)


def test_repository_saves_and_revokes_provider_approval(tmp_path: Path) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    approval = ThirdPartyProcessingApproval(
        approval_id=uuid4(),
        provider_id="provider-cn-1",
        endpoint_region="cn-shanghai",
        data_scope=MediaAssetKind.KEYFRAME,
        purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        cross_border=False,
        cross_border_assessment="DOMESTIC_ONLY",
        max_cost_cny="2.50",
        approved_by="reviewer-1",
        approved_at=NOW,
        approval_evidence_sha256="b" * 64,
    )

    repository.create_provider_approval(approval)
    assert repository.find_active_provider_approval("provider-cn-1") == approval
    repository.revoke_provider_approval(
        approval.approval_id,
        revoked_at=NOW,
        revocation_reason="provider disabled",
    )
    assert repository.find_active_provider_approval("provider-cn-1") is None


@pytest.mark.parametrize("content", ["not-json", '{"schema_version":"9.0"}'])
def test_repository_corruption_fails_closed(tmp_path: Path, content: str) -> None:
    directory = tmp_path / "access"
    directory.mkdir(mode=0o700)
    store = directory / "store.json"
    store.write_text(content, encoding="utf-8")
    store.chmod(0o600)

    with pytest.raises(AccessRepositoryError):
        JsonAccessRepository(store).list_events()


def test_concurrent_event_append_preserves_complete_unique_prefix(tmp_path: Path) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    session_id = uuid4()

    with ThreadPoolExecutor(max_workers=8) as pool:
        stored = tuple(pool.map(repository.append_event, [_event(session_id) for _ in range(32)]))

    events = repository.list_events()
    assert len(events) == 32
    assert [event.sequence for event in events] == list(range(1, 33))
    assert len({event.event_id for event in events}) == 32
    assert {event.event_id for event in events} == {event.event_id for event in stored}


@pytest.mark.parametrize("failure", ["permissions", "symlink", "oversized", "too-deep"])
def test_repository_rejects_unsafe_or_bounded_input(tmp_path: Path, failure: str) -> None:
    directory = tmp_path / "access"
    directory.mkdir(mode=0o700)
    store = directory / "store.json"
    if failure == "permissions":
        store.write_text("{}", encoding="utf-8")
        store.chmod(0o644)
    elif failure == "symlink":
        target = tmp_path / "outside.json"
        target.write_text("{}", encoding="utf-8")
        store.symlink_to(target)
    elif failure == "oversized":
        store.write_bytes(b" " * (MAX_ACCESS_STORE_BYTES + 1))
        store.chmod(0o600)
    else:
        nested: object = "leaf"
        for _ in range(40):
            nested = {"next": nested}
        store.write_text(json.dumps(nested), encoding="utf-8")
        store.chmod(0o600)

    with pytest.raises(AccessRepositoryError):
        JsonAccessRepository(store).list_events()


def test_repository_rejects_reordered_or_overwritten_event_prefix(tmp_path: Path) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    session_id = uuid4()
    repository.append_event(_event(session_id))
    repository.append_event(_event(session_id))
    payload = json.loads(repository.path.read_text(encoding="utf-8"))
    payload["events"][0]["sequence"] = 2
    repository.path.write_text(json.dumps(payload), encoding="utf-8")
    repository.path.chmod(0o600)

    with pytest.raises(AccessRepositoryError):
        repository.list_events()


def test_repository_rejects_wrong_owner_boundary(tmp_path: Path, monkeypatch) -> None:
    repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    repository.append_event(_event(uuid4()))
    monkeypatch.setattr(
        "billiards_ai.cloud.privacy.access_repository.os.geteuid", lambda: 2**31 - 1
    )

    with pytest.raises(AccessRepositoryError):
        repository.list_events()
