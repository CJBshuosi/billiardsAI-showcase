import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from billiards_ai.cloud.privacy.retention_repository import (
    JsonRetentionRepository,
    MediaUnavailableError,
    RetentionConflictError,
    RetentionNotFoundError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    AnonymizationRecord,
    AssetContext,
    BystanderDisposition,
    DeletionRequestStatus,
    DeletionScope,
    MaterialUsePurpose,
    MediaAssetKind,
    MediaDeletionStatus,
    RetainedMediaAsset,
    RetainedMediaAssetCreate,
    SourceAssetProvenance,
    SubjectClassification,
)

NOW = datetime(2026, 7, 27, 18, tzinfo=UTC)


def _asset(*, session_id=None, captured_at=NOW, sources=()) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=session_id or uuid4(),
        owner_user_id="user-1",
        kind=MediaAssetKind.FRAME if sources else MediaAssetKind.VIDEO,
        storage_locator=f"critical/{uuid4()}.bin",
        content_sha256=uuid4().hex * 2,
        source_asset_ids=sources,
        captured_at=captured_at,
        created_at=NOW,
        updated_at=NOW,
    )


def _context(asset: RetainedMediaAsset, approved) -> AssetContext:
    return AssetContext(
        asset_id=str(asset.asset_id),
        asset_sha256=asset.content_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=asset.captured_at,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )


def _create(asset: RetainedMediaAsset) -> RetainedMediaAssetCreate:
    return RetainedMediaAssetCreate.model_validate(
        asset.model_dump(
            include={
                "asset_id",
                "session_id",
                "owner_scope",
                "kind",
                "storage_locator",
                "content_sha256",
                "captured_at",
                "source_asset_ids",
            }
        )
    )


def _register(service, asset: RetainedMediaAsset, approved) -> RetainedMediaAsset:
    return service.register_asset(
        _create(asset),
        gate_revision_id=approved.revision_id,
        asset_context=_context(asset, approved),
        requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
    )


def test_received_session_scope_rejects_late_independent_video(
    repo, tmp_path, approved_gate_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    service = MediaRetentionService(repo, gate_repository=gate_service.repository, now=lambda: NOW)
    first = _asset()
    _register(service, first, approved)
    service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(session_ids=(first.session_id,)),
        idempotency_key="received-session-tombstone",
    )

    with pytest.raises(MediaUnavailableError):
        _register(service, _asset(session_id=first.session_id), approved)


def test_completion_window_cannot_commit_completed_with_late_active_asset(
    repo, tmp_path, approved_gate_factory, controlled_media_store_factory, monkeypatch
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    media = controlled_media_store_factory(tmp_path)
    service = MediaRetentionService(
        repo,
        gate_repository=gate_service.repository,
        media_store=media,
        now=lambda: NOW,
    )
    first = _asset()
    _register(service, first, approved)
    target = media.root / first.storage_locator
    target.parent.mkdir(mode=0o700, parents=True)
    target.write_bytes(b"synthetic")
    target.chmod(0o600)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(session_ids=(first.session_id,)),
        idempotency_key="completion-linearization",
    )
    late = _asset(session_id=first.session_id)
    real_finalize = repo.finalize_completed_request
    injected = False
    blocked = False

    def inject_before_completion(candidate):
        nonlocal injected, blocked
        if not injected:
            injected = True
            with pytest.raises(MediaUnavailableError):
                _register(service, late, approved)
            blocked = True
        return real_finalize(candidate)

    monkeypatch.setattr(repo, "finalize_completed_request", inject_before_completion)
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.status is DeletionRequestStatus.COMPLETED
    assert blocked is True
    with pytest.raises(RetentionNotFoundError):
        repo.get_asset(late.asset_id)


def test_completed_all_user_media_rejects_old_capture_late_registration(
    repo, tmp_path, approved_gate_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    service = MediaRetentionService(repo, gate_repository=gate_service.repository, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(all_user_media=True),
        idempotency_key="all-media-tombstone",
    )
    assert request.status is DeletionRequestStatus.COMPLETED
    late = _asset(captured_at=request.created_at - timedelta(seconds=1))

    with pytest.raises(MediaUnavailableError):
        _register(service, late, approved)


@pytest.mark.parametrize("mismatch", ["hash", "captured_at"])
def test_gate_context_must_match_actual_retention_asset(
    repo, tmp_path, approved_gate_factory, mismatch
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    service = MediaRetentionService(repo, gate_repository=gate_service.repository, now=lambda: NOW)
    item = _asset()
    context = _context(item, approved)
    if mismatch == "hash":
        context = context.model_copy(update={"asset_sha256": "f" * 64})
    else:
        context = context.model_copy(
            update={"captured_at": item.captured_at + timedelta(seconds=1)}
        )

    with pytest.raises((RetentionConflictError, MediaUnavailableError)):
        service.register_asset(
            _create(item),
            gate_revision_id=approved.revision_id,
            asset_context=context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )
    assert repo.list_assets() == ()
    assert item.deletion_status is MediaDeletionStatus.ACTIVE


def test_gate_source_provenance_must_match_registered_retention_source(
    repo, tmp_path, approved_gate_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    service = MediaRetentionService(repo, gate_repository=gate_service.repository, now=lambda: NOW)
    source = _asset()
    _register(service, source, approved)
    derived = _asset(session_id=source.session_id, sources=(source.asset_id,))
    fake_source_hash = "e" * 64
    context = AssetContext(
        asset_id=str(derived.asset_id),
        asset_sha256=derived.content_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=derived.captured_at,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=True,
        classifications=(
            SubjectClassification(
                asset_id=str(derived.asset_id),
                segment_ref="whole",
                subject_ref="person-1",
                disposition=BystanderDisposition.ANONYMIZED,
            ),
        ),
        anonymization_records=(
            AnonymizationRecord(
                source_asset_id=str(source.asset_id),
                source_asset_sha256=fake_source_hash,
                derived_asset_id=str(derived.asset_id),
                method="blur",
                operator_id="operator-1",
                processed_at=NOW,
                output_sha256=derived.content_sha256,
                visual_evidence_sha256="a" * 64,
                metadata_removal_evidence_sha256="b" * 64,
            ),
        ),
        source_asset_provenance=SourceAssetProvenance(
            asset_id=str(source.asset_id),
            sha256=fake_source_hash,
            captured_at=source.captured_at,
            gate_revision_id=approved.revision_id,
            gate_scope_fingerprint=approved.scope.fingerprint,
        ),
    )

    with pytest.raises((RetentionConflictError, MediaUnavailableError)):
        service.register_asset(
            _create(derived),
            gate_revision_id=approved.revision_id,
            asset_context=context,
            requested_purpose=MaterialUsePurpose.SERVICE_ANALYSIS,
        )


def test_registration_dto_cannot_preseed_lifecycle_or_milestones() -> None:
    create_type = RetainedMediaAssetCreate
    values = {
        "asset_id": uuid4(),
        "session_id": uuid4(),
        "owner_user_id": "user-1",
        "kind": MediaAssetKind.VIDEO,
        "storage_locator": "dto/video.mp4",
        "content_sha256": "c" * 64,
        "captured_at": NOW,
    }
    for forged in (
        {"deletion_status": MediaDeletionStatus.DELETED},
        {"deletion_status": MediaDeletionStatus.PENDING_DELETION},
        {"reviewed_at": NOW},
        {"retention_expires_at": NOW + timedelta(days=30)},
    ):
        with pytest.raises(ValidationError):
            create_type(**values, **forged)


def test_repository_has_no_public_gate_bypass_or_locator_resolver() -> None:
    assert not hasattr(JsonRetentionRepository, "add_asset")
    assert not hasattr(JsonRetentionRepository, "resolve_active_locator")


def test_production_modules_do_not_call_test_seed_or_bypass_authorized_registration() -> None:
    forbidden_calls: list[str] = []
    for source in Path("src").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "_insert_asset_for_testing":
                forbidden_calls.append(f"{source}:{node.lineno}:test-seed")
            if (
                node.func.attr == "_register_authorized_asset"
                and source.name != "retention_service.py"
            ):
                forbidden_calls.append(f"{source}:{node.lineno}:gate-bypass")
    assert forbidden_calls == []
