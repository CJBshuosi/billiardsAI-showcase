from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from billiards_ai.cloud.privacy.access_repository import JsonAccessRepository
from billiards_ai.cloud.privacy.access_service import AccessDeniedError, MediaAccessService
from billiards_ai.cloud.privacy.media_store import ControlledFilesystemMediaStore
from billiards_ai.cloud.privacy.retention_repository import (
    MediaUnavailableError,
    RetentionConflictError,
    RetentionNotFoundError,
    RetentionRepositoryError,
)
from billiards_ai.cloud.privacy.retention_service import MediaRetentionService
from billiards_ai.shared.models import (
    AccessAction,
    AccessActorRole,
    AccessDecisionResult,
    AssetContext,
    DeletionFailureCode,
    DeletionRequest,
    DeletionRequestStatus,
    DeletionScope,
    MediaAssetKind,
    MediaDeletionStatus,
    RetainedMediaAsset,
    ReviewerIdentity,
)

NOW = datetime(2026, 7, 27, 18, tzinfo=UTC)


def _put_synthetic(media, locator: str, content: bytes = b"synthetic") -> Path:
    path = media.root / locator
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def make_asset(*, session_id, locator: str, kind, sources=()) -> RetainedMediaAsset:
    return RetainedMediaAsset(
        asset_id=uuid4(),
        session_id=session_id,
        owner_user_id="user-1",
        kind=kind,
        storage_locator=locator,
        content_sha256=uuid4().hex * 2,
        source_asset_ids=sources,
        captured_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def test_execute_deletes_transitive_derivatives_and_scrubs_reverse_references(
    repo, tmp_path: Path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    session_id = uuid4()
    video = make_asset(
        session_id=session_id, locator="session/video.mp4", kind=MediaAssetKind.VIDEO
    )
    frame = make_asset(
        session_id=session_id,
        locator="session/frame.jpg",
        kind=MediaAssetKind.FRAME,
        sources=(video.asset_id,),
    )
    keyframe = make_asset(
        session_id=session_id,
        locator="session/keyframe.jpg",
        kind=MediaAssetKind.KEYFRAME,
        sources=(video.asset_id,),
    )
    screenshot = make_asset(
        session_id=session_id,
        locator="session/screenshot.jpg",
        kind=MediaAssetKind.SCREENSHOT,
        sources=(frame.asset_id,),
    )
    derivative = make_asset(
        session_id=session_id,
        locator="session/visualization.png",
        kind=MediaAssetKind.TRACEABLE_DERIVATIVE,
        sources=(keyframe.asset_id, screenshot.asset_id),
    )
    all_media = (video, frame, keyframe, screenshot, derivative)
    for item in all_media:
        repo._insert_asset_for_testing(item)
        _put_synthetic(media, item.storage_locator)
    repo.register_aggregate(
        "report-1",
        "user-1",
        {
            "score": 83,
            "video_url": f"private://{video.asset_id}",
            "source_sha256": video.content_sha256,
            "metrics": {"accuracy": 0.91, "preview": video.storage_locator},
        },
        (video.asset_id, screenshot.asset_id, derivative.asset_id),
        media_ref_paths=("/video_url", "/source_sha256", "/metrics/preview"),
    )
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(video.asset_id,)),
        idempotency_key="delete-video-tree",
    )
    assert request.frozen_asset_ids == tuple(sorted((item.asset_id for item in all_media), key=str))
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.status is DeletionRequestStatus.COMPLETED
    assert len(completed.attempts) == 1
    assert all(
        repo.get_asset(item.asset_id).deletion_status is MediaDeletionStatus.DELETED
        for item in all_media
    )
    with pytest.raises(MediaUnavailableError):
        repo._get_active_asset_with_locator(video.asset_id)
    assert repo.get_aggregate("report-1") == {
        "owner_user_id": "user-1",
        "data": {"score": 83, "metrics": {"accuracy": 0.91}},
        "media_asset_ids": [],
        "media_ref_paths": [],
    }


def test_filesystem_store_rejects_path_escape_and_symlinks(
    tmp_path: Path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    with pytest.raises(RetentionRepositoryError):
        media.delete("../outside")
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not delete")
    link = media.root / "link"
    link.symlink_to(outside)
    with pytest.raises(RetentionRepositoryError):
        media.delete("link")
    assert outside.exists()


def test_media_root_requires_existing_private_sentinel_without_chmod_side_effect(
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "someones-photos"
    unrelated.mkdir(mode=0o755)
    unrelated.chmod(0o755)
    with pytest.raises(RetentionRepositoryError):
        ControlledFilesystemMediaStore(unrelated)
    assert unrelated.stat().st_mode & 0o777 == 0o755

    unrelated.chmod(0o700)
    with pytest.raises(RetentionRepositoryError, match="sentinel"):
        ControlledFilesystemMediaStore(unrelated)
    assert not hasattr(ControlledFilesystemMediaStore, "put_for_phase0")


def test_media_root_rejects_home_and_its_first_level_children(tmp_path: Path, monkeypatch) -> None:
    synthetic_home = tmp_path / "home"
    photos = synthetic_home / "Photos"
    photos.mkdir(parents=True, mode=0o700)
    photos.chmod(0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: synthetic_home))
    with pytest.raises(RetentionRepositoryError, match="broad"):
        ControlledFilesystemMediaStore(photos)


def test_failed_attempt_is_visible_retryable_and_preserved_on_retry(
    repo, tmp_path: Path, controlled_media_store_factory
) -> None:
    class FailOnceStore(ControlledFilesystemMediaStore):
        should_fail = True

        def delete(self, locator: str) -> None:
            if self.should_fail:
                raise RetentionRepositoryError("synthetic storage failure")
            super().delete(locator)

    media = controlled_media_store_factory(tmp_path, FailOnceStore)
    item = make_asset(session_id=uuid4(), locator="session/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="retry-me",
    )
    failed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert failed.status is DeletionRequestStatus.FAILED
    assert failed.retryable is True
    assert (
        service.get_public_status(
            current_user_id="user-1", request_id=request.request_id
        ).next_action
        == "RETRY_OR_CONTACT_OPERATOR"
    )
    media.should_fail = False
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.status is DeletionRequestStatus.COMPLETED
    assert [attempt.result for attempt in completed.attempts] == ["FAILED", "COMPLETED"]


def test_completed_media_cannot_be_resolved_after_an_allowing_gate(
    repo, tmp_path, approved_gate_factory, controlled_media_store_factory
) -> None:
    gate_service, _, approved = approved_gate_factory(tmp_path)
    media = controlled_media_store_factory(tmp_path)
    item = make_asset(session_id=uuid4(), locator="session/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    retention_service = MediaRetentionService(
        repo,
        media_store=media,
        gate_repository=gate_service.repository,
        now=lambda: NOW,
    )
    context = AssetContext(
        asset_id=str(item.asset_id),
        asset_sha256=item.content_sha256,
        gate_scope_fingerprint=approved.scope.fingerprint,
        captured_at=NOW,
        captured_under_revision_id=approved.revision_id,
        bystanders_present=False,
    )
    identity = ReviewerIdentity(
        principal_id="reviewer-1",
        roles=(AccessActorRole.ACCESS_ADMIN, AccessActorRole.REVIEWER),
        credential_id="synthetic-reviewer",
    )
    identity_provider = type(
        "FixedIdentityProvider", (), {"current_identity": lambda self: identity}
    )()
    access_repository = JsonAccessRepository(tmp_path / "access" / "store.json")
    access_service = MediaAccessService(
        access_repository=access_repository,
        retention_service=retention_service,
        identity_provider=identity_provider,
        now=lambda: NOW,
    )
    access_service.grant_current_reviewer(session_id=item.session_id, reason="assigned")
    observed = []
    access_service.authorize_and_record_access(
        asset_id=item.asset_id,
        gate_revision_id=approved.revision_id,
        asset_context=context,
        reason="manual review",
        action=AccessAction.CONTROLLED_REVIEW_VIEW,
        consumer=observed.append,
    )
    assert observed == [item.storage_locator]
    assert access_repository.list_events()[-1].result is AccessDecisionResult.ALLOW

    request = retention_service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="access-denied-after-delete",
    )
    retention_service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    with pytest.raises(AccessDeniedError):
        access_service.authorize_and_record_access(
            asset_id=item.asset_id,
            gate_revision_id=approved.revision_id,
            asset_context=context,
            reason="manual review retry",
            action=AccessAction.CONTROLLED_REVIEW_VIEW,
            consumer=lambda _: pytest.fail("deleted media must not be released"),
        )
    denied = access_repository.list_events()[-1]
    assert denied.result is AccessDecisionResult.DENY
    assert denied.security_event is True


def test_execute_recloses_derivatives_registered_after_submission(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    session_id = uuid4()
    video = make_asset(session_id=session_id, locator="window/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(video)
    _put_synthetic(media, video.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(video.asset_id,)),
        idempotency_key="window-race",
    )
    assert request.frozen_asset_ids == (video.asset_id,)

    late_frame = make_asset(
        session_id=session_id,
        locator="window/late-frame.jpg",
        kind=MediaAssetKind.FRAME,
        sources=(video.asset_id,),
    )
    repo._insert_asset_for_testing(late_frame)
    _put_synthetic(media, late_frame.storage_locator)

    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.status is DeletionRequestStatus.COMPLETED
    assert completed.frozen_asset_ids == tuple(
        sorted((video.asset_id, late_frame.asset_id), key=str)
    )
    assert repo.get_asset(late_frame.asset_id).deletion_status is MediaDeletionStatus.DELETED
    assert not media.exists(late_frame.storage_locator)


def test_completion_fails_if_derivative_appears_during_physical_deletion(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    session_id = uuid4()
    video = make_asset(session_id=session_id, locator="during/video.mp4", kind=MediaAssetKind.VIDEO)
    late_frame = make_asset(
        session_id=session_id,
        locator="during/late-frame.jpg",
        kind=MediaAssetKind.FRAME,
        sources=(video.asset_id,),
    )

    class RegisterDuringDeleteStore(ControlledFilesystemMediaStore):
        injected = False

        def delete(self, locator: str) -> None:
            super().delete(locator)
            if not self.injected:
                self.injected = True
                repo._insert_asset_for_testing(late_frame)
                _put_synthetic(self, late_frame.storage_locator)

    media = controlled_media_store_factory(tmp_path, RegisterDuringDeleteStore)
    repo._insert_asset_for_testing(video)
    _put_synthetic(media, video.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(video.asset_id,)),
        idempotency_key="during-delete-race",
    )
    failed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert failed.status is DeletionRequestStatus.FAILED
    assert failed.failure_reason_code.value == "VERIFICATION_FAILED"
    assert repo.get_asset(late_frame.asset_id).deletion_status is MediaDeletionStatus.ACTIVE


def test_keyboard_interrupt_records_failed_attempt_and_remains_retryable(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    class InterruptingStore(ControlledFilesystemMediaStore):
        def delete(self, locator: str) -> None:
            raise KeyboardInterrupt("synthetic Ctrl-C")

    media = controlled_media_store_factory(tmp_path, InterruptingStore)
    item = make_asset(session_id=uuid4(), locator="interrupt/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="interrupt",
    )
    with pytest.raises(KeyboardInterrupt):
        service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    failed = repo.get_request(request.request_id)
    assert failed.status is DeletionRequestStatus.FAILED
    assert len(failed.attempts) == 1
    assert [event.event_type.value for event in failed.events] == ["STARTED"]
    assert failed.retryable is True


def test_explicit_reclaim_recovers_stranded_in_progress_request(repo) -> None:
    item = make_asset(session_id=uuid4(), locator="reclaim/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    service = MediaRetentionService(repo, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="reclaim",
    )
    stranded = DeletionRequest.model_validate(
        request.model_copy(update={"status": DeletionRequestStatus.IN_PROGRESS}).model_dump()
    )
    repo.save_request(stranded)
    reclaimed = service.reclaim_in_progress(
        request_id=request.request_id,
        executor_id="Boyu",
        reason="operator confirmed previous process is gone",
    )
    assert reclaimed.status is DeletionRequestStatus.FAILED
    assert reclaimed.retryable is True
    assert reclaimed.attempts[-1].planned_asset_count == 1
    assert reclaimed.events[-1].event_type.value == "RECLAIMED"


def test_aggregate_registration_fails_closed_on_unindexed_refs_then_scrubs_declared_refs(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    item = make_asset(session_id=uuid4(), locator="scrub/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    with pytest.raises(RetentionConflictError):
        repo.register_aggregate(
            "adversarial-report",
            "user-1",
            {
                "score": 80,
                "frame_count": 42,
                "cdn_link": f"https://cdn.example/{item.asset_id}.mp4",
                "notes": [f"see {item.storage_locator}"],
                str(item.asset_id): {"trim": "0-5s"},
                "provenance": {"digest": item.content_sha256},
            },
            (item.asset_id,),
            media_ref_paths=("/provenance/digest",),
        )

    repo.register_aggregate(
        "adversarial-report",
        "user-1",
        {
            "score": 80,
            "frame_count": 42,
            "cdn_link": f"https://cdn.example/{item.asset_id}.mp4",
            "notes": [f"see {item.storage_locator}"],
            str(item.asset_id): {"trim": "0-5s"},
            "provenance": {"digest": item.content_sha256},
        },
        (item.asset_id,),
        media_ref_paths=(
            "/cdn_link",
            "/notes/0",
            f"/{item.asset_id}",
            "/provenance/digest",
        ),
    )
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="adversarial-scrub",
    )
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.status is DeletionRequestStatus.COMPLETED
    assert repo.get_aggregate("adversarial-report") == {
        "owner_user_id": "user-1",
        "data": {"score": 80, "frame_count": 42, "notes": [], "provenance": {}},
        "media_asset_ids": [],
        "media_ref_paths": [],
    }


def test_completion_evidence_counts_preexisting_absent_objects(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    session_id = uuid4()
    present = make_asset(
        session_id=session_id, locator="evidence/present.mp4", kind=MediaAssetKind.VIDEO
    )
    already_absent = make_asset(
        session_id=session_id,
        locator="evidence/already-absent.jpg",
        kind=MediaAssetKind.FRAME,
        sources=(present.asset_id,),
    )
    repo._insert_asset_for_testing(present)
    repo._insert_asset_for_testing(already_absent)
    _put_synthetic(media, present.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(present.asset_id,)),
        idempotency_key="real-evidence-counts",
    )
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert completed.attempts[-1].verified_deleted_count == 2
    assert "preexisting_absent=1" in completed.attempts[-1].evidence_summary


def test_asset_load_failure_keeps_frozen_planned_count(
    repo, tmp_path, controlled_media_store_factory, monkeypatch
) -> None:
    media = controlled_media_store_factory(tmp_path)
    item = make_asset(
        session_id=uuid4(), locator="evidence/load-fail.mp4", kind=MediaAssetKind.VIDEO
    )
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="load-failure-count",
    )
    monkeypatch.setattr(
        repo,
        "get_asset",
        lambda asset_id: (_ for _ in ()).throw(RetentionRepositoryError("synthetic load")),
    )
    failed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    assert failed.status is DeletionRequestStatus.FAILED
    assert failed.attempts[-1].planned_asset_count == 1


def test_concurrent_execution_allows_only_one_executor(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    entered = Event()
    release = Event()

    class BlockingStore(ControlledFilesystemMediaStore):
        def delete(self, locator: str) -> None:
            entered.set()
            assert release.wait(timeout=5)
            super().delete(locator)

    media = controlled_media_store_factory(tmp_path, BlockingStore)
    item = make_asset(session_id=uuid4(), locator="concurrent/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="single-executor",
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            service.execute_deletion,
            request_id=request.request_id,
            executor_id="operator-1",
        )
        assert entered.wait(timeout=5)
        with pytest.raises(RetentionConflictError):
            service.execute_deletion(request_id=request.request_id, executor_id="operator-2")
        release.set()
        assert first.result(timeout=5).status is DeletionRequestStatus.COMPLETED


def test_non_owner_query_and_completed_rollback_fail_closed(
    repo, tmp_path, controlled_media_store_factory
) -> None:
    media = controlled_media_store_factory(tmp_path)
    item = make_asset(session_id=uuid4(), locator="terminal/video.mp4", kind=MediaAssetKind.VIDEO)
    repo._insert_asset_for_testing(item)
    _put_synthetic(media, item.storage_locator)
    service = MediaRetentionService(repo, media_store=media, now=lambda: NOW)
    request = service.submit_deletion(
        current_user_id="user-1",
        scope=DeletionScope(asset_ids=(item.asset_id,)),
        idempotency_key="terminal",
    )
    with pytest.raises(RetentionNotFoundError, match="not found"):
        service.get_public_status(current_user_id="user-2", request_id=request.request_id)
    completed = service.execute_deletion(request_id=request.request_id, executor_id="Boyu")
    rollback = DeletionRequest.model_validate(
        completed.model_copy(
            update={
                "status": DeletionRequestStatus.FAILED,
                "completed_at": None,
                "completion_evidence_summary": None,
                "failure_reason_code": DeletionFailureCode.INVALID_STATE,
                "retryable": False,
            }
        ).model_dump()
    )
    with pytest.raises(RetentionConflictError):
        repo.save_request(rollback)
