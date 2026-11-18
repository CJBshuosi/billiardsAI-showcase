from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from billiards_ai.edge.route_assistance.control import WristbandRole
from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.postgame import SavedMediaOutboxItem
from billiards_ai.edge.route_assistance.records import (
    LiveShotFact,
    LiveShotFactOutboxItem,
)
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    HttpResult,
    SyncAttemptResult,
    SyncMediaItem,
    SyncOutboxJob,
    SyncRequestToSign,
    UrllibSyncHttpTransport,
    _normalize_windows_credential_blob,
    classify_http_status,
    sign_sync_request,
)


def test_urllib_transport_can_disable_redirect_following(monkeypatch) -> None:
    handlers = []

    class Response:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://development.invalid/v1/test"
            assert timeout == 20.0
            return Response()

    def build_opener(*received):
        handlers.extend(received)
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    transport = UrllibSyncHttpTransport("https://development.invalid", allow_redirects=False)

    assert transport.post_json("/v1/test", {}, b"{}").status_code == 200
    assert transport._allow_redirects is False
    assert len(handlers) == 1
    assert handlers[0].redirect_request(None, None, 302, "Found", {}, "new") is None


def _store(path: Path, now: datetime) -> SQLiteFieldRecordStore:
    return SQLiteFieldRecordStore(
        path,
        now_utc=lambda: now,
        session_id_factory=lambda: "session-1",
    )


def test_reopening_database_preserves_due_sync_job(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 14, 0, tzinfo=UTC)
    path = tmp_path / "field.sqlite3"
    store = _store(path, now)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session("session-1")
    job = store.enqueue_sync_job(
        session_id="session-1",
        manifest_hash="a" * 64,
        manifest_json='{"schema_version":"1.0"}',
    )

    reopened = _store(path, now)
    assert reopened.load_sync_job(job.sync_job_id).status == "pending"
    claimed = reopened.claim_due_sync_job(now)
    assert claimed is not None
    assert claimed.sync_job_id == job.sync_job_id
    assert claimed.status == "claimed"


def test_live_shot_fact_lost_ack_retries_same_key_without_uploading_media() -> None:
    fact = LiveShotFact.create(
        session_id="session-1",
        shot_id="shot-0001",
        sequence=1,
        start_frame_id=100,
        settle_frame_id=220,
        mode="observation",
        actor_role=WristbandRole.RED,
        actor_user_id="user-red",
        observed_outcome="unknown",
        rule_revision=3,
        recording_reference="session-recording:epoch-1:100-220",
    )
    payload = json.dumps(asdict(fact), default=str, sort_keys=True, separators=(",", ":"))
    item = LiveShotFactOutboxItem(fact, payload, "leased", 1, None)

    class Credentials:
        def device_id(self):
            return "device-1"

        def secret(self):
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.requests = []

        def put_json(self, path, headers, body):
            self.requests.append((path, headers, body))
            if len(self.requests) == 1:
                raise OSError("ack lost")
            return HttpResult(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "session_id": fact.session_id,
                            "shot_id": fact.shot_id,
                            "idempotency_key": fact.idempotency_key,
                        },
                    }
                ).encode(),
            )

        def post_json(self, *_args, **_kwargs):
            raise AssertionError("live shot facts must not use session media sync")

        def put_file(self, *_args, **_kwargs):
            raise AssertionError("live shot facts must not upload video")

    transport = Transport()
    client = FieldSyncClient(credentials=Credentials(), transport=transport)

    assert client.run_live_shot_fact_once(item).outcome == "retry"
    assert client.run_live_shot_fact_once(item).outcome == "complete"
    assert [request[0] for request in transport.requests] == [
        "/v1/field-sessions/session-1/shots/shot-0001",
        "/v1/field-sessions/session-1/shots/shot-0001",
    ]
    assert {request[1]["x-idempotency-key"] for request in transport.requests} == {
        fact.idempotency_key
    }
    assert transport.requests[0][2] == transport.requests[1][2]


def test_partial_retry_is_durable_and_never_persists_credentials_or_urls(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 26, 14, 0, tzinfo=UTC)
    store = _store(tmp_path / "field.sqlite3", now)
    store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session("session-1")
    job = store.enqueue_sync_job(
        session_id="session-1",
        manifest_hash="a" * 64,
        manifest_json='{"schema_version":"1.0"}',
    )
    store.claim_due_sync_job(now)
    retry_at = now + timedelta(minutes=1)
    partial = store.mark_sync_partial(
        job.sync_job_id,
        error_code="NETWORK_UNAVAILABLE",
        next_attempt_at=retry_at,
    )

    assert partial.status == "partial"
    assert partial.retry_count == 1
    assert store.claim_due_sync_job(now) is None
    claimed = store.claim_due_sync_job(retry_at)
    assert claimed is not None
    assert claimed.sync_job_id == partial.sync_job_id
    assert claimed.status == "claimed"
    raw = (tmp_path / "field.sqlite3").read_bytes()
    assert b"fixture-secret" not in raw
    assert b"https://upload.invalid" not in raw


def test_session_scoped_claim_never_drains_another_ended_session(tmp_path: Path) -> None:
    now = datetime(2026, 8, 26, 14, 0, tzinfo=UTC)
    session_ids = iter(("session-1", "session-2"))
    sync_ids = iter(("sync-2", "sync-1"))
    store = SQLiteFieldRecordStore(
        tmp_path / "field.sqlite3",
        now_utc=lambda: now,
        session_id_factory=session_ids.__next__,
        sync_job_id_factory=sync_ids.__next__,
    )
    first = store.open_or_resume_session(artifact_id="artifact-1")
    store.end_session(first.session_id)
    second = store.open_or_resume_session(artifact_id="artifact-2")
    store.end_session(second.session_id)
    second_job = store.enqueue_sync_job(
        session_id=second.session_id,
        manifest_hash="b" * 64,
        manifest_json='{"session":"two"}',
    )
    first_job = store.enqueue_sync_job(
        session_id=first.session_id,
        manifest_hash="a" * 64,
        manifest_json='{"session":"one"}',
    )

    claimed_first = store.claim_due_sync_job_for_session(first.session_id, now)
    claimed_second = store.claim_due_sync_job_for_session(second.session_id, now)

    assert claimed_first is not None and claimed_first.sync_job_id == first_job.sync_job_id
    assert claimed_second is not None and claimed_second.sync_job_id == second_job.sync_job_id


def test_signature_covers_method_path_time_nonce_and_body_hash() -> None:
    request = SyncRequestToSign(
        method="POST",
        path="/v1/sync/jobs",
        timestamp="1700000000",
        nonce="nonce-0001",
        body_sha256="a" * 64,
    )
    signed = sign_sync_request(request, secret=b"fixture-secret")
    assert signed.signature == ("6b9e4a82381b1691779c82688592c2e199f8ba8ae3baa97c8adbf4f43d2d756f")


def test_windows_credential_blob_normalizes_cmdkey_unicode_password() -> None:
    secret = "0123456789abcdef" * 4

    assert _normalize_windows_credential_blob(secret.encode("utf-16-le")) == secret.encode()


def test_windows_credential_blob_preserves_application_defined_utf8_bytes() -> None:
    secret = b"0123456789abcdef" * 4

    assert _normalize_windows_credential_blob(secret) == secret


def test_retry_classifier_only_retries_transient_failures() -> None:
    assert classify_http_status(408) == "retry"
    assert classify_http_status(429) == "retry"
    assert classify_http_status(503) == "retry"
    assert classify_http_status(401) == "blocked"
    assert classify_http_status(409) == "blocked"
    assert classify_http_status(422) == "blocked"
    assert classify_http_status(200) == "success"


def test_sync_client_signs_create_and_complete_without_leaking_secret() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.requests: list[tuple[str, dict[str, str], bytes]] = []

        def post_json(self, path: str, headers: dict[str, str], body: bytes) -> HttpResult:
            self.requests.append((path, headers, body))
            if path == "/v1/sync/jobs":
                return HttpResult(200, b'{"data":{"job_id":"cloud-job-1"}}')
            return HttpResult(200, b'{"data":{"status":"completed"}}')

        def put_file(self, url: str, path: Path, content_type: str) -> HttpResult:
            raise AssertionError((url, path, content_type))

    transport = Transport()
    client = FieldSyncClient(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(("nonce-create-0001", "nonce-complete-01")).__next__,
    )
    result = client.run_once(
        SyncOutboxJob(
            sync_job_id="sync-1",
            session_id="session-1",
            manifest_hash="a" * 64,
            manifest_json='{"schema_version":"1.0"}',
            status="claimed",
            retry_count=0,
            next_attempt_at_utc="2026-08-26T14:00:00Z",
            last_error_code=None,
        )
    )

    assert result.outcome == "complete"
    assert [request[0] for request in transport.requests] == [
        "/v1/sync/jobs",
        "/v1/sync/jobs/cloud-job-1/complete",
    ]
    assert "fixture-secret" not in repr(transport.requests)


def test_sync_client_attests_immediately_before_each_mutation_and_blocks_unknown_env() -> None:
    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def post_json(self, path, _headers, _body):
            self.calls.append(path)
            return HttpResult(200, b'{"data":{"job_id":"cloud-job-1"}}')

    transport = Transport()

    def reject_unknown_environment() -> None:
        raise OSError("development runtime attestation did not match")

    result = FieldSyncClient(
        credentials=Credentials(),
        transport=transport,
        before_mutation=reject_unknown_environment,
    ).run_once(
        SyncOutboxJob(
            sync_job_id="sync-1",
            session_id="session-1",
            manifest_hash="a" * 64,
            manifest_json='{"schema_version":"1.0"}',
            status="claimed",
            retry_count=0,
            next_attempt_at_utc="2026-08-26T14:00:00Z",
            last_error_code=None,
        )
    )

    assert result == SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
    assert transport.calls == []


def test_sync_client_forwards_private_upload_authorization_headers(tmp_path: Path) -> None:
    clip = tmp_path / "shot-1.mp4"
    clip.write_bytes(b"private-video")

    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.upload: tuple[str, Path, str, dict[str, str]] | None = None

        def post_json(self, path: str, headers: dict[str, str], body: bytes) -> HttpResult:
            if path == "/v1/sync/jobs":
                return HttpResult(200, b'{"data":{"job_id":"cloud-job-1"}}')
            if path.endswith("/media-authorizations"):
                return HttpResult(
                    200,
                    b'{"data":{"method":"PUT","url":"https://cos.invalid/upload","headers":{"authorization":"signed","x-cos-security-token":"token","x-cos-meta-fileid":"cos-file","key":"sessions%2Fsession-1%2Fclip-1.mp4"}}}',
                )
            return HttpResult(200, b'{"data":{"status":"completed"}}')

        def put_file(
            self,
            url: str,
            path: Path,
            content_type: str,
            headers: dict[str, str],
        ) -> HttpResult:
            self.upload = (url, path, content_type, headers)
            return HttpResult(200, b"")

    transport = Transport()
    result = FieldSyncClient(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(("nonce-create", "nonce-authorize", "nonce-complete")).__next__,
    ).run_once(
        SyncOutboxJob(
            sync_job_id="sync-1",
            session_id="session-1",
            manifest_hash="a" * 64,
            manifest_json='{"schema_version":"1.0"}',
            status="claimed",
            retry_count=0,
            next_attempt_at_utc="2026-08-26T14:00:00Z",
            last_error_code=None,
            media=(
                SyncMediaItem(
                    clip_id="clip-1",
                    shot_id="shot-1",
                    local_path=clip,
                    content_type="video/mp4",
                    content_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
                    byte_size=clip.stat().st_size,
                ),
            ),
        )
    )

    assert result.outcome == "complete"
    assert transport.upload == (
        "https://cos.invalid/upload",
        clip,
        "video/mp4",
        {
            "authorization": "signed",
            "x-cos-security-token": "token",
            "x-cos-meta-fileid": "cos-file",
            "key": "sessions%2Fsession-1%2Fclip-1.mp4",
        },
    )


def test_sync_client_treats_an_already_completed_idempotent_job_as_complete(
    tmp_path: Path,
) -> None:
    clip = tmp_path / "shot-1.mp4"
    clip.write_bytes(b"private-video")

    class Credentials:
        def device_id(self) -> str:
            return "device-1"

        def secret(self) -> bytes:
            return b"fixture-secret"

    class Transport:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def post_json(self, path: str, headers: dict[str, str], body: bytes) -> HttpResult:
            self.paths.append(path)
            return HttpResult(
                200,
                b'{"data":{"job_id":"cloud-job-1","status":"completed"}}',
            )

        def put_file(
            self,
            url: str,
            path: Path,
            content_type: str,
            headers: dict[str, str],
        ) -> HttpResult:
            raise AssertionError((url, path, content_type, headers))

    transport = Transport()
    result = FieldSyncClient(credentials=Credentials(), transport=transport).run_once(
        SyncOutboxJob(
            sync_job_id="sync-1",
            session_id="session-1",
            manifest_hash="a" * 64,
            manifest_json='{"schema_version":"1.0"}',
            status="claimed",
            retry_count=1,
            next_attempt_at_utc="2026-08-26T14:00:00Z",
            last_error_code="NETWORK_UNAVAILABLE",
            media=(
                SyncMediaItem(
                    clip_id="clip-1",
                    shot_id="shot-1",
                    local_path=clip,
                    content_type="video/mp4",
                    content_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
                    byte_size=clip.stat().st_size,
                ),
            ),
        )
    )

    assert result.outcome == "complete"
    assert result.error_code is None
    assert transport.paths == ["/v1/sync/jobs"]


def _saved_item(root: Path) -> tuple[SavedMediaOutboxItem, bytes]:
    payload = b"validated-saved-video"
    media = root / "postgame" / "session-1" / "clip.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(payload)
    from billiards_ai.edge.route_assistance.records import (
        ownership_receipt_from_payload,
        ownership_receipt_sha256,
    )

    fixture = json.loads(
        (
            Path(__file__).parents[3]
            / "fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json"
        ).read_text()
    )
    ownership = ownership_receipt_from_payload(fixture["descriptor"]["ownership_receipt"])
    item = SavedMediaOutboxItem(
        clip_id="clip-af6c58ceb2457aa4e77276b7dfa43360ca31fb0d100a8bc74a5cc7ce40851084",
        session_id="session-1",
        shot_id="shot-0001",
        postgame_manifest_sha256="b" * 64,
        job_id="job-1",
        artifact_id="c" * 64,
        local_path=media,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        content_type="video/mp4",
        source_kind="usb_live",
        start_ms=3_000,
        end_ms=5_000,
        duration_ms=2_000,
        owner_user_id=ownership.actor_user_id,
        actor_role=ownership.actor_role,
        ownership_receipt_sha256=ownership_receipt_sha256(ownership),
        ownership_receipt=ownership,
        save_request_id="save-1",
        save_receipt_sha256="d" * 64,
        status="leased",
        attempts=1,
        claim_token="claim-1",
        lease_until_utc="2026-08-26T14:01:00+00:00",
        next_attempt_at_utc="2026-08-26T14:00:00+00:00",
        last_error_code=None,
    )
    return item, payload


class _SavedCredentials:
    def device_id(self) -> str:
        return "device-1"

    def secret(self) -> bytes:
        return b"fixture-secret"


def _saved_ack(item: SavedMediaOutboxItem, *, status: str, job_id: str = "saved-job-1") -> bytes:
    return json.dumps(
        {
            "data": {
                "job_id": job_id,
                "status": status,
                "clip_id": item.clip_id,
                "session_id": item.session_id,
                "shot_id": item.shot_id,
                "content_sha256": item.content_sha256,
                "publication_kind": "saved_postgame_clip",
            }
        },
        separators=(",", ":"),
    ).encode()


@pytest.mark.parametrize(
    "upload_attempt",
    [
        None,
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789abc",
            "sequence": 1,
        },
    ],
)
def test_saved_clip_attempt_signs_exact_descriptor_and_uploads_validated_bytes(
    tmp_path: Path,
    upload_attempt,
) -> None:
    item, original = _saved_item(tmp_path)

    class Transport:
        def __init__(self) -> None:
            self.posts: list[tuple[str, dict[str, str], bytes]] = []
            self.upload: tuple[str, bytes, str, dict[str, str]] | None = None

        def post_json(self, path, headers, body):
            self.posts.append((path, headers, body))
            if path == "/v1/sync/saved-clips":
                return HttpResult(200, _saved_ack(item, status="pending"))
            if path.endswith("/media-authorization"):
                item.local_path.write_bytes(b"replaced-after-validation")
                return HttpResult(
                    200,
                    json.dumps(
                        {
                            "data": {
                                "method": "PUT",
                                "url": "https://upload.invalid/object?signature=secret",
                                "headers": {
                                    "authorization": "signed",
                                    "x-cos-security-token": "token",
                                },
                                **(
                                    {"upload_attempt": upload_attempt}
                                    if upload_attempt is not None
                                    else {}
                                ),
                            }
                        }
                    ).encode(),
                )
            return HttpResult(200, _saved_ack(item, status="published"))

        def put_bytes(self, url, payload, content_type, headers):
            self.upload = (url, payload, content_type, headers)
            return HttpResult(200, b"")

        def put_json(self, *_args):
            raise AssertionError("saved clips must not use the live-fact route")

        def put_file(self, *_args):
            raise AssertionError("saved clips must upload the already validated bytes")

    transport = Transport()
    client = FieldSyncClient(
        credentials=_SavedCredentials(),
        transport=transport,
        now_seconds=lambda: 1_700_000_000,
        nonce_factory=iter(("nonce-create", "nonce-authorize", "nonce-complete")).__next__,
    )

    result = client.run_saved_media_once(
        item,
        trusted_postgame_root=tmp_path / "postgame",
    )

    assert result == SyncAttemptResult("complete")
    assert [request[0] for request in transport.posts] == [
        "/v1/sync/saved-clips",
        "/v1/sync/saved-clips/saved-job-1/media-authorization",
        "/v1/sync/saved-clips/saved-job-1/complete",
    ]
    fixture_path = (
        Path(__file__).parents[3] / "fixtures/synthetic/miniprogram/saved-postgame-clip-e2e-v2.json"
    )
    expected = json.loads(fixture_path.read_text())["descriptor"]
    expected.update(
        artifact_id="c" * 64,
        byte_size=21,
        clip_id="clip-af6c58ceb2457aa4e77276b7dfa43360ca31fb0d100a8bc74a5cc7ce40851084",
        content_sha256="3990f4a6d022c5f21f01d26d1644d40f0c22232b2b674a1c4d8addb99478a0d6",
        start_ms=3000,
        end_ms=5000,
        duration_ms=2000,
        save_request_id="save-1",
        save_receipt_sha256="d" * 64,
    )
    expected_body = json.dumps(
        expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert transport.posts[0][2] == expected_body
    assert transport.posts[0][1] == {
        "content-type": "application/json",
        "x-device-id": "device-1",
        "x-timestamp": "1700000000",
        "x-nonce": "nonce-create",
        "x-body-sha256": hashlib.sha256(expected_body).hexdigest(),
        "x-signature": sign_sync_request(
            SyncRequestToSign(
                method="POST",
                path="/v1/sync/saved-clips",
                timestamp="1700000000",
                nonce="nonce-create",
                body_sha256=hashlib.sha256(expected_body).hexdigest(),
            ),
            secret=b"fixture-secret",
        ).signature,
        "x-idempotency-key": item.clip_id,
    }
    assert transport.upload == (
        "https://upload.invalid/object?signature=secret",
        original,
        "video/mp4",
        {"authorization": "signed", "x-cos-security-token": "token"},
    )

    expected_completion = {} if upload_attempt is None else {"upload_attempt": upload_attempt}
    completion_path, completion_headers, completion_body = transport.posts[-1]
    assert json.loads(completion_body) == expected_completion
    assert completion_headers["x-body-sha256"] == hashlib.sha256(completion_body).hexdigest()
    assert (
        completion_headers["x-signature"]
        == sign_sync_request(
            SyncRequestToSign(
                method="POST",
                path=completion_path,
                timestamp="1700000000",
                nonce="nonce-complete",
                body_sha256=hashlib.sha256(completion_body).hexdigest(),
            ),
            secret=b"fixture-secret",
        ).signature
    )


@pytest.mark.parametrize(
    "upload_attempt",
    [
        None,
        {},
        [],
        True,
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789abc",
            "sequence": True,
        },
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789abc",
            "sequence": "1",
        },
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789abc",
            "sequence": 65,
        },
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789ABC",
            "sequence": 1,
        },
        {
            "contract": "managed_upload_attempt_v1",
            "operation_id": "12345678-1234-4234-8234-123456789abc",
            "sequence": 1,
            "epoch": "private",
        },
    ],
)
def test_saved_clip_malformed_attempt_blocks_before_put(tmp_path, upload_attempt):
    item, _ = _saved_item(tmp_path)

    class Transport:
        def post_json(self, path, headers, body):
            if path == "/v1/sync/saved-clips":
                return HttpResult(200, _saved_ack(item, status="pending"))
            assert path.endswith("/media-authorization")
            return HttpResult(
                200,
                json.dumps(
                    {
                        "data": {
                            "method": "PUT",
                            "url": "https://upload.invalid",
                            "headers": {},
                            "upload_attempt": upload_attempt,
                        }
                    }
                ).encode(),
            )

        def put_bytes(self, *_args):
            raise AssertionError("malformed attempt must not PUT")

    result = FieldSyncClient(
        credentials=_SavedCredentials(), transport=Transport()
    ).run_saved_media_once(item, trusted_postgame_root=tmp_path / "postgame")
    assert result == SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")


def test_saved_media_stop_budget_covers_complete_attested_client_attempt(tmp_path, monkeypatch):
    import threading
    from urllib.parse import urlsplit

    from billiards_ai.lab.field_assistance.cloud_gateway import (
        DevelopmentAttestedFieldCloudGateway,
        SignedFieldCloudGateway,
    )
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundSavedMediaSyncWorker

    item, original = _saved_item(tmp_path)
    requests = []
    responses = {
        "/v1/runtime-attestation": json.dumps(
            {
                "ok": True,
                "data": {
                    "schema_version": "1.0",
                    "runtime_environment": "development",
                    "cloudbase_environment_id": "development-fixture",
                },
            }
        ).encode(),
        "/v1/sync/saved-clips": _saved_ack(item, status="pending"),
        "/v1/sync/saved-clips/saved-job-1/media-authorization": (
            b'{"data":{"method":"PUT","url":"https://upload.invalid/object","headers":{}}}'
        ),
        "/object": b"",
        "/v1/sync/saved-clips/saved-job-1/complete": _saved_ack(item, status="published"),
    }

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Opener:
        def open(self, request, timeout):
            path = urlsplit(request.full_url).path
            requests.append((request.method, path, timeout))
            if path == "/object":
                assert request.data == original
            return Response(responses[path])

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    transport = UrllibSyncHttpTransport("https://development.invalid", allow_redirects=False)
    credentials = _SavedCredentials()
    gateway = DevelopmentAttestedFieldCloudGateway(
        delegate=SignedFieldCloudGateway(credentials=credentials, transport=transport),
        expected_cloudbase_environment_id="development-fixture",
    )
    client = FieldSyncClient(
        credentials=credentials,
        transport=transport,
        before_mutation=gateway.runtime_attestation,
    )
    worker = _BackgroundSavedMediaSyncWorker(
        store=object(),
        client=client,
        trusted_postgame_root=tmp_path / "postgame",
        wake_event=threading.Event(),
    )

    class InFlightAttempt:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            # Run the actual complete attempt; model each request using its full
            # transport timeout without network access or wall-clock sleeps.
            assert client.run_saved_media_once(
                item, trusted_postgame_root=tmp_path / "postgame"
            ) == SyncAttemptResult("complete")
            self.alive = timeout < sum(request[2] for request in requests)
            assert timeout >= sum(request[2] for request in requests) + 5.0

    worker._thread = InFlightAttempt()
    worker.stop()

    assert not worker._thread.is_alive()
    assert requests == [
        ("POST", "/v1/runtime-attestation", 20.0),
        ("POST", "/v1/sync/saved-clips", 20.0),
        ("POST", "/v1/runtime-attestation", 20.0),
        ("POST", "/v1/sync/saved-clips/saved-job-1/media-authorization", 20.0),
        ("POST", "/v1/runtime-attestation", 20.0),
        ("PUT", "/object", 20.0),
        ("POST", "/v1/runtime-attestation", 20.0),
        ("POST", "/v1/sync/saved-clips/saved-job-1/complete", 20.0),
    ]


def test_saved_clip_lost_ack_replays_same_body_and_accepts_bound_completion(
    tmp_path: Path,
) -> None:
    item, _ = _saved_item(tmp_path)

    class Transport:
        def __init__(self) -> None:
            self.creates: list[tuple[dict[str, str], bytes]] = []

        def post_json(self, path, headers, body):
            assert path == "/v1/sync/saved-clips"
            self.creates.append((headers, body))
            if len(self.creates) == 1:
                raise OSError("ack lost")
            return HttpResult(200, _saved_ack(item, status="completed"))

        def put_bytes(self, *_args):
            raise AssertionError("a completed replay must not upload again")

    transport = Transport()
    client = FieldSyncClient(credentials=_SavedCredentials(), transport=transport)

    assert client.run_saved_media_once(
        item, trusted_postgame_root=tmp_path / "postgame"
    ) == SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
    assert client.run_saved_media_once(
        item, trusted_postgame_root=tmp_path / "postgame"
    ) == SyncAttemptResult("complete")
    assert transport.creates[0][1] == transport.creates[1][1]
    assert {request[0]["x-idempotency-key"] for request in transport.creates} == {item.clip_id}


def test_saved_clip_accepts_cloud_job_identity_allowed_by_route_contract(
    tmp_path: Path,
) -> None:
    item, _ = _saved_item(tmp_path)

    class Transport:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def post_json(self, path, _headers, _body):
            self.paths.append(path)
            if path == "/v1/sync/saved-clips":
                return HttpResult(
                    200,
                    _saved_ack(item, status="pending", job_id="saved.job-1"),
                )
            return HttpResult(503, b'{"error":{"code":"SYNC_PARTIAL"}}')

    transport = Transport()
    result = FieldSyncClient(
        credentials=_SavedCredentials(), transport=transport
    ).run_saved_media_once(item, trusted_postgame_root=tmp_path / "postgame")

    assert result == SyncAttemptResult("retry", "SYNC_PARTIAL")
    assert transport.paths == [
        "/v1/sync/saved-clips",
        "/v1/sync/saved-clips/saved.job-1/media-authorization",
    ]


def test_saved_clip_retries_when_device_credential_is_temporarily_unavailable(
    tmp_path: Path,
) -> None:
    item, _ = _saved_item(tmp_path)

    class Credentials(_SavedCredentials):
        def secret(self) -> bytes:
            raise OSError("credential manager unavailable")

    class Transport:
        def post_json(self, *_args):
            raise AssertionError("a missing credential must block cloud mutation")

    result = FieldSyncClient(credentials=Credentials(), transport=Transport()).run_saved_media_once(
        item, trusted_postgame_root=tmp_path / "postgame"
    )

    assert result == SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")


@pytest.mark.parametrize(
    ("status_code", "error_code", "expected"),
    [
        (503, "SYNC_PARTIAL", SyncAttemptResult("retry", "SYNC_PARTIAL")),
        (409, "SYNC_PARTIAL", SyncAttemptResult("retry", "SYNC_PARTIAL")),
        (409, "SYNC_IDENTITY_CONFLICT", SyncAttemptResult("blocked", "SYNC_IDENTITY_CONFLICT")),
    ],
)
def test_saved_clip_create_classifies_partial_and_identity_failures(
    tmp_path: Path,
    status_code: int,
    error_code: str,
    expected: SyncAttemptResult,
) -> None:
    item, _ = _saved_item(tmp_path)

    class Transport:
        def post_json(self, path, _headers, _body):
            assert path == "/v1/sync/saved-clips"
            return HttpResult(status_code, json.dumps({"error": {"code": error_code}}).encode())

    assert (
        FieldSyncClient(
            credentials=_SavedCredentials(), transport=Transport()
        ).run_saved_media_once(item, trusted_postgame_root=tmp_path / "postgame")
        == expected
    )


@pytest.mark.parametrize("attack", ["outside", "symlink", "ancestor_symlink", "digest", "size"])
def test_saved_clip_blocks_unsafe_or_replaced_media_before_cloud_mutation(
    tmp_path: Path,
    attack: str,
) -> None:
    item, _ = _saved_item(tmp_path)
    trusted = tmp_path / "postgame"
    if attack == "outside":
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(item.local_path.read_bytes())
        item = replace(item, local_path=outside)
    elif attack == "symlink":
        target = item.local_path.with_name("target.mp4")
        target.write_bytes(item.local_path.read_bytes())
        item.local_path.unlink()
        item.local_path.symlink_to(target)
    elif attack == "ancestor_symlink":
        real = tmp_path / "real"
        real.mkdir()
        target = real / "clip.mp4"
        target.write_bytes(item.local_path.read_bytes())
        link = trusted / "linked"
        link.symlink_to(real, target_is_directory=True)
        item = replace(item, local_path=link / "clip.mp4")
    elif attack == "digest":
        item.local_path.write_bytes(b"wrong digest but size")
        item = replace(item, byte_size=item.local_path.stat().st_size)
    else:
        item = replace(item, byte_size=item.byte_size + 1)

    class Transport:
        def post_json(self, *_args):
            raise AssertionError("unsafe media must be blocked before cloud mutation")

    result = FieldSyncClient(
        credentials=_SavedCredentials(), transport=Transport()
    ).run_saved_media_once(item, trusted_postgame_root=trusted)

    assert result == SyncAttemptResult("blocked", "SYNC_LOCAL_MEDIA_INVALID")


def test_saved_clip_blocks_above_maximum_before_read_or_cloud_mutation(
    tmp_path: Path,
) -> None:
    item, _ = _saved_item(tmp_path)
    object.__setattr__(item, "byte_size", 268_435_457)

    class Transport:
        def post_json(self, *_args):
            raise AssertionError("oversized media must be blocked before cloud mutation")

    result = FieldSyncClient(
        credentials=_SavedCredentials(), transport=Transport()
    ).run_saved_media_once(item, trusted_postgame_root=tmp_path / "postgame")

    assert result == SyncAttemptResult("blocked", "SYNC_MEDIA_TOO_LARGE")


@pytest.mark.parametrize(
    ("authorization", "expected_code"),
    [
        (
            {"method": "POST", "url": "https://upload.invalid", "headers": {}},
            "SYNC_INVALID_RESPONSE",
        ),
        (
            {"method": "PUT", "url": "http://upload.invalid", "headers": {}},
            "SYNC_INVALID_RESPONSE",
        ),
        (
            {"method": "PUT", "url": "https://user:secret@upload.invalid", "headers": {}},
            "SYNC_INVALID_RESPONSE",
        ),
        (
            {
                "method": "PUT",
                "url": "https://upload.invalid",
                "headers": {"x-ok": "bad\r\nvalue"},
            },
            "SYNC_INVALID_RESPONSE",
        ),
    ],
)
def test_saved_clip_rejects_unsafe_upload_authorization(
    tmp_path: Path,
    authorization: dict[str, object],
    expected_code: str,
) -> None:
    item, _ = _saved_item(tmp_path)

    class Transport:
        def post_json(self, path, _headers, _body):
            if path == "/v1/sync/saved-clips":
                return HttpResult(200, _saved_ack(item, status="pending"))
            return HttpResult(
                200,
                json.dumps({"data": authorization}, separators=(",", ":")).encode(),
            )

        def put_bytes(self, *_args):
            raise AssertionError("unsafe authorization must not upload")

    result = FieldSyncClient(
        credentials=_SavedCredentials(), transport=Transport()
    ).run_saved_media_once(item, trusted_postgame_root=tmp_path / "postgame")

    assert result == SyncAttemptResult("blocked", expected_code)
