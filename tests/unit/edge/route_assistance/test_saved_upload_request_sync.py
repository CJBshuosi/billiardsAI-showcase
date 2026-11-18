"""Managed single-use transport recovery, with real synthetic SQLite history."""
# Imported fixtures are injected by their public names.
# ruff: noqa: F811

import hashlib
import hmac
import json
import socket
import sqlite3
import threading
from dataclasses import replace
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from billiards_ai.edge.route_assistance.persistence import SQLiteFieldRecordStore
from billiards_ai.edge.route_assistance.saved_upload_requests import SavedUploadRequestLedger
from billiards_ai.edge.route_assistance.sync import (
    FieldSyncClient,
    HttpResult,
    SyncAttemptResult,
    UrllibSyncHttpTransport,
)
from tests.unit.edge.route_assistance.test_local_owner_media_cleanup import media  # noqa: F401
from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot
from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
    NOW,
    SCOPE,
    owner_module,
    payload,
)
from tests.unit.edge.route_assistance.test_saved_upload_requests import attempt, scene  # noqa: F401


class Credentials:
    def device_id(self):
        return "device-1"

    def secret(self):
        return b"synthetic-device-secret"


def rows(store):
    with sqlite3.connect(store.path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM field_saved_upload_requests")]


def ack(item, status):
    return HttpResult(
        200,
        json.dumps(
            {
                "data": dict(
                    job_id="cloud-job-1",
                    status=status,
                    clip_id=item.clip_id,
                    session_id=item.session_id,
                    shot_id=item.shot_id,
                    content_sha256=item.content_sha256,
                    publication_kind="saved_postgame_clip",
                )
            }
        ).encode(),
    )


class Transport:
    def __init__(self, store, item):
        self.store, self.item = store, item
        self.calls = []
        self.upload_attempt = attempt()
        self.authorization_requests = []
        self.grant_edit = lambda data: None
        self.on_authorize = lambda: None
        self.on_put = lambda: None
        self.put_result = HttpResult(200, b"")
        self.complete_loss = False
        self.authorize_loss = False

    def post_json(self, path, headers, body):
        self.calls.append(path)
        # The real HMAC/nonces must cover each exact serialized request body.
        digest = hashlib.sha256(body).hexdigest()
        canonical = (
            f"POST\n{path}\n{headers['x-timestamp']}\n{headers['x-nonce']}\n{digest}".encode()
        )
        assert headers["x-body-sha256"] == digest
        assert (
            headers["x-signature"]
            == hmac.new(Credentials().secret(), canonical, hashlib.sha256).hexdigest()
        )
        if path == "/v1/sync/saved-clips":
            assert body == self.item.descriptor_json.encode()
            assert headers["x-idempotency-key"] == self.item.idempotency_key
            return ack(self.item, "pending")
        if path.endswith("/media-authorization"):
            row = rows(self.store)[-1]
            assert row["state"] == "requested"
            request = {"contract": "managed_upload_request_v1", "request_id": row["request_id"]}
            assert json.loads(body) == {"upload_request": request}
            self.authorization_requests.append(request)
            self.on_authorize()
            if self.authorize_loss:
                self.authorize_loss = False
                raise OSError("lost authorization response with secret")
            data = dict(
                method="PUT",
                url="https://synthetic.invalid/secret-url",
                headers={"authorization": "secret-authorization"},
                expires_in_seconds=300,
                upload_attempt=self.upload_attempt,
                upload_request=request,
                request_status="issued",
            )
            self.grant_edit(data)
            return HttpResult(200, json.dumps({"data": data}).encode())
        assert path == "/v1/sync/saved-clips/cloud-job-1/complete"
        row = rows(self.store)[-1]
        assert row["state"] == "succeeded" and row["terminal_sha256"]
        assert json.loads(body) == {"upload_attempt": self.upload_attempt}
        if self.complete_loss:
            self.complete_loss = False
            raise OSError("lost complete ACK")
        return ack(self.item, "published")

    def put_bytes_once(self, url, payload, content_type, headers):
        self.calls.append("PUT")
        assert rows(self.store)[-1]["state"] == "started"
        assert rows(self.store)[-1]["upload_attempt_json"] == json.dumps(
            self.upload_attempt, sort_keys=True, separators=(",", ":")
        )
        assert url == "https://synthetic.invalid/secret-url"
        assert headers == {"authorization": "secret-authorization"}
        assert hashlib.sha256(payload).hexdigest() == self.item.content_sha256
        assert len(payload) == self.item.byte_size and content_type == self.item.content_type
        self.on_put()
        if isinstance(self.put_result, BaseException):
            raise self.put_result
        return self.put_result

    def put_bytes(self, *_):
        raise AssertionError("managed transport must never fall back to ordinary PUT")


def run(scene, transport, before_mutation=lambda: None):
    store, item = scene
    return FieldSyncClient(
        credentials=Credentials(), transport=transport, before_mutation=before_mutation
    ).run_saved_media_once(
        item,
        trusted_postgame_root=store.artifact_path_for_session(item.session_id) / "postgame",
        upload_requests=SavedUploadRequestLedger(store, SCOPE),
    )


def test_single_put_records_before_network_and_keeps_secrets_out_of_ledger(scene):
    store, item = scene
    transport = Transport(store, item)
    original = item.local_path.read_bytes()
    transport.on_authorize = lambda: item.local_path.write_bytes(b"changed-after-validation")
    assert run(scene, transport) == SyncAttemptResult("complete")
    assert transport.calls == [
        "/v1/sync/saved-clips",
        "/v1/sync/saved-clips/cloud-job-1/media-authorization",
        "PUT",
        "/v1/sync/saved-clips/cloud-job-1/complete",
    ]
    encoded = json.dumps(rows(store))
    for secret in (
        "secret-url",
        "secret-authorization",
        "synthetic-device-secret",
        str(item.local_path),
    ):
        assert secret not in encoded
    assert rows(store)[0]["state"] == "succeeded"
    assert hashlib.sha256(original).hexdigest() == item.content_sha256


@pytest.mark.parametrize("status", [201, 204, 302, 400, 408, 500])
def test_non200_is_unknown_and_blocks_before_even_completed_create(scene, status):
    transport = Transport(*scene)
    transport.put_result = HttpResult(status, b"secret provider detail")
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert rows(scene[0])[0]["state"] == "unknown"
    transport.calls.clear()
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert transport.calls == []


@pytest.mark.parametrize("error", [OSError("response lost"), RuntimeError("provider failure")])
def test_put_exception_never_retries_or_reauthorizes(scene, error):
    transport = Transport(*scene)
    transport.put_result = error
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert transport.calls.count("PUT") == 1
    assert rows(scene[0])[0]["state"] == "unknown"


def test_crash_after_start_blocks_before_network(scene):
    with SavedUploadRequestLedger(scene[0], SCOPE).use(scene[1]) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
        session.record_started(row.request_id)
    transport = Transport(*scene)
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert transport.calls == [] and rows(scene[0])[0]["state"] == "unknown"


def test_received_restart_retires_then_allocates_fresh_on_later_attempt(scene):
    with SavedUploadRequestLedger(scene[0], SCOPE).use(scene[1]) as session:
        old = session.load_or_create("cloud-job-1")
        session.record_received(old.request_id, attempt())
    transport = Transport(*scene)
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_UPLOAD_REQUEST_RETIRED")
    assert transport.calls == [] and rows(scene[0])[0]["state"] == "retired"
    assert run(scene, transport) == SyncAttemptResult("complete")
    assert rows(scene[0])[1]["request_id"] != old.request_id


def test_lost_authorization_retries_same_uuid_and_reserved_never_puts(scene):
    transport = Transport(*scene)
    transport.authorize_loss = True
    assert run(scene, transport) == SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
    assert rows(scene[0])[0]["state"] == "requested"

    def reserve(data):
        for field in ("url", "headers", "method", "expires_in_seconds"):
            del data[field]
        data["request_status"] = "reserved"

    transport.grant_edit = reserve
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_UPLOAD_REQUEST_RETIRED")
    assert transport.authorization_requests[0] == transport.authorization_requests[1]
    assert "PUT" not in transport.calls
    assert rows(scene[0])[0]["state"] == "retired"
    transport.grant_edit = lambda data: None
    assert run(scene, transport) == SyncAttemptResult("complete")
    assert transport.authorization_requests[-1] != transport.authorization_requests[0]


def test_success_lost_complete_ack_reopens_and_only_replays_exact_complete_without_file(scene):
    transport = Transport(*scene)
    transport.complete_loss = True
    assert run(scene, transport) == SyncAttemptResult("retry", "NETWORK_UNAVAILABLE")
    first = rows(scene[0])[0]
    scene[1].local_path.unlink()
    reopened = SQLiteFieldRecordStore(scene[0].path, now_utc=lambda: NOW)
    transport.calls.clear()
    assert run((reopened, scene[1]), transport) == SyncAttemptResult("complete")
    assert transport.calls == ["/v1/sync/saved-clips/cloud-job-1/complete"]
    assert rows(reopened)[0] == first


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.pop("upload_request"),
        lambda data: data["upload_request"].update(
            request_id="11111111-1111-4111-8111-111111111111"
        ),
        lambda data: data["upload_request"].update(extra=True),
        lambda data: data.update(request_status="other"),
        lambda data: data.pop("upload_attempt"),
        lambda data: data["upload_attempt"].update(sequence=True),
        lambda data: data.update(extra="secret"),
        lambda data: data.update(
            request_status="reserved"
        ),  # reserved must never contain credentials
        lambda data: data.update(expires_in_seconds=True),
    ],
)
def test_managed_response_is_exact_and_never_starts_malformed_grant(scene, change):
    transport = Transport(*scene)
    transport.grant_edit = change
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_INVALID_RESPONSE")
    assert "PUT" not in transport.calls and rows(scene[0])[0]["state"] == "requested"


def test_missing_single_use_transport_fails_before_start(scene):
    transport = Transport(*scene)
    transport.put_bytes_once = None
    assert run(scene, transport) == SyncAttemptResult(
        "blocked", "SYNC_UPLOAD_TRANSPORT_UNAVAILABLE"
    )
    assert all(row["state"] not in {"started", "succeeded"} for row in rows(scene[0]))
    assert "PUT" not in transport.calls


@pytest.mark.parametrize("terminal", ["succeeded", "unknown"])
def test_failed_terminal_commit_retains_started_and_reports_unknown(scene, terminal):
    store, item = scene
    transport = Transport(*scene)
    if terminal == "unknown":
        transport.put_result = HttpResult(500, b"")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER deny_terminal BEFORE UPDATE ON field_saved_upload_requests "
            f"WHEN NEW.state='{terminal}' BEGIN SELECT RAISE(ABORT,'synthetic fault'); END"
        )
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert rows(store)[0]["state"] == "started"
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER deny_terminal")
    transport.calls.clear()
    assert run(scene, transport) == SyncAttemptResult("blocked", "SYNC_UPLOAD_OUTCOME_UNKNOWN")
    assert rows(store)[0]["state"] == "unknown" and transport.calls == []


@pytest.mark.parametrize("expiry_call, state", [(1, None), (3, "received")])
def test_attestation_then_atomic_current_claim_check_fences_create_and_start(
    scene, expiry_call, state
):
    store, _ = scene
    transport = Transport(*scene)
    attestations = []

    def attest():
        attestations.append(True)
        if len(attestations) == expiry_call:
            store._now_utc = lambda: NOW + timedelta(seconds=2)

    assert run(scene, transport, attest) == SyncAttemptResult(
        "retry", "SYNC_SAVED_MEDIA_STALE_CLAIM"
    )
    assert "PUT" not in transport.calls
    assert (rows(store)[0]["state"] if rows(store) else None) == state
    if expiry_call == 1:
        assert transport.calls == []


def test_stale_received_claim_cannot_retire_or_create_another_request(scene):
    store, item = scene
    with SavedUploadRequestLedger(store, SCOPE).use(item) as session:
        row = session.load_or_create("cloud-job-1")
        session.record_received(row.request_id, attempt())
    replacement = store.claim_due_saved_media(NOW + timedelta(seconds=2), managed_scope=SCOPE)
    assert replacement.claim_token != item.claim_token
    transport = Transport(*scene)
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_SAVED_MEDIA_STALE_CLAIM")
    assert rows(store)[0]["state"] == "received" and transport.calls == []


def test_owner_delete_during_put_retains_success_and_fences_complete(scene):
    store, _ = scene
    transport = Transport(*scene)
    transport.on_put = lambda: store.accept_owner_delete_task(
        owner_module().OwnerDeleteDeviceTask.from_payload(payload(store, shot="shot-0002")),
        managed_scope=SCOPE,
    )
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_SAVED_MEDIA_STALE_CLAIM")
    assert rows(store)[0]["state"] == "succeeded"
    assert not any(path.endswith("/complete") for path in transport.calls)


def test_sync_preserves_original_facts_ownership_descriptors_and_receipts(scene):
    before = snapshot(scene[0].path)
    assert run(scene, Transport(*scene)) == SyncAttemptResult("complete")
    after = snapshot(scene[0].path)
    changed_tables = {"field_saved_upload_requests", "field_upload_receipt_outbox"}
    assert {k: v for k, v in before.items() if k not in changed_tables} == {
        k: v for k, v in after.items() if k not in changed_tables
    }


def test_wrong_source_descriptor_blocks_before_network(scene):
    transport = Transport(*scene)
    assert run((scene[0], replace(scene[1], content_sha256="f" * 64)), transport) == (
        SyncAttemptResult("blocked", "SYNC_LOCAL_MEDIA_INVALID")
    )
    assert transport.calls == []


def test_pin_contention_is_retryable_and_makes_no_network_call(scene):
    with SavedUploadRequestLedger(scene[0], SCOPE).use(scene[1]):
        transport = Transport(*scene)
        assert run(scene, transport) == SyncAttemptResult(
            "retry", "SYNC_UPLOAD_REQUEST_UNAVAILABLE"
        )
        assert transport.calls == []


@pytest.mark.parametrize("reserved", [False, True])
def test_claim_reclaimed_during_authorization_cannot_start_or_retire(scene, reserved):
    store, item = scene
    transport = Transport(*scene)

    def reclaim():
        replacement = store.claim_due_saved_media(NOW + timedelta(seconds=2), managed_scope=SCOPE)
        assert replacement.claim_token != item.claim_token
        store._now_utc = lambda: NOW + timedelta(seconds=2)

    transport.on_authorize = reclaim
    if reserved:

        def reserve(data):
            for field in ("url", "headers", "method", "expires_in_seconds"):
                del data[field]
            data["request_status"] = "reserved"

        transport.grant_edit = reserve
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_SAVED_MEDIA_STALE_CLAIM")
    assert "PUT" not in transport.calls and rows(store)[0]["state"] == "requested"


def test_terminal_success_survives_claim_expiry_but_complete_is_fenced(scene):
    store, item = scene
    transport = Transport(*scene)
    transport.on_put = lambda: setattr(store, "_now_utc", lambda: NOW + timedelta(seconds=2))
    assert run(scene, transport) == SyncAttemptResult("retry", "SYNC_SAVED_MEDIA_STALE_CLAIM")
    assert rows(store)[0]["state"] == "succeeded"
    assert not any(path.endswith("/complete") for path in transport.calls)
    replacement = store.claim_due_saved_media(NOW + timedelta(seconds=2), managed_scope=SCOPE)
    transport.calls.clear()
    assert run((store, replacement), transport) == SyncAttemptResult("complete")
    assert transport.calls == ["/v1/sync/saved-clips/cloud-job-1/complete"]


def test_managed_worker_uses_original_root_and_real_scope_bound_ledger(media):
    from billiards_ai.lab.field_assistance.cloud_runtime import _BackgroundSavedMediaSyncWorker

    store, _, root, *_ = media

    class Client:
        def run_saved_media_once(self, item, *, trusted_postgame_root, upload_requests):
            from billiards_ai.edge.route_assistance.local_media_lock import session_media_lock

            assert trusted_postgame_root == root / "postgame"
            assert isinstance(upload_requests, SavedUploadRequestLedger)
            with upload_requests.use(item) as session:
                with (
                    pytest.raises(TimeoutError),
                    session_media_lock(root, item.session_id, exclusive=True),
                ):
                    pytest.fail("worker must retain original media pin while using ledger")
                session.require_current_claim()
                session.load_or_create("cloud-job-1")
            return SyncAttemptResult("retry", "CONTROLLED_RETRY")

    worker = _BackgroundSavedMediaSyncWorker(
        store=store,
        client=Client(),
        trusted_postgame_root=root / "wrong-current-root",
        wake_event=threading.Event(),
        now_utc=lambda: NOW,
        managed_scope=SCOPE,
    )
    assert worker.run_once()
    assert len(rows(store)) == 1 and rows(store)[0]["state"] == "requested"


@pytest.mark.parametrize("status", [200, 201, 307, 500])
def test_production_once_put_returns_terminal_response_without_redirect_or_retry(status):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            calls.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            if status == 307:
                self.send_header("Location", "/second-write")
            self.end_headers()
            self.wfile.write(b"terminal")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        transport = UrllibSyncHttpTransport(url)  # even default legacy redirect policy
        assert transport.put_bytes_once(
            url + "/first", b"synthetic", "video/mp4", {}
        ) == HttpResult(status, b"terminal")
        assert calls == [("/first", b"synthetic")]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_production_once_put_does_not_retry_response_loss():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            calls.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with pytest.raises(OSError):
            UrllibSyncHttpTransport(url).put_bytes_once(url, b"synthetic", "video/mp4", {})
        assert calls == [b"synthetic"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
