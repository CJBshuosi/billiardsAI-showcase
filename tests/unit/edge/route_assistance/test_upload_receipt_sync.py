"""Actual receipt signing and worker settlement against synthetic retained SQLite."""

# ruff: noqa: F811
import hashlib
import hmac
import importlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from billiards_ai.edge.route_assistance.sync import FieldSyncClient, HttpResult, SyncAttemptResult
from tests.unit.edge.route_assistance.test_upload_receipt_delivery import (
    NOW,  # noqa: F401
    SCOPE,
    ack,
    media,  # noqa: F401
    outbox,
    rows,
    scene,  # noqa: F401
    terminal,
)


class Credentials:
    def device_id(self):
        return SCOPE.device_id

    def secret(self):
        return b"synthetic-secret"


class Transport:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post_json(self, path, headers, body):
        self.calls.append((path, headers, body))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def client(item, response=None, **kwargs):
    transport = Transport(response or HttpResult(200, json.dumps({"data": ack(item)}).encode()))
    nonces = iter(["fresh-nonce-0001", "fresh-nonce-0002", "fresh-nonce-0003"])
    return FieldSyncClient(
        credentials=Credentials(),
        transport=transport,
        now_seconds=lambda: 1793491200,
        nonce_factory=lambda: next(nonces),
        **kwargs,
    ), transport


def worker(store, client, **kwargs):
    module = importlib.import_module("billiards_ai.lab.field_assistance.cloud_runtime")
    return module._BackgroundUploadReceiptSyncWorker(
        store=store, client=client, managed_scope=SCOPE, now_utc=lambda: NOW, **kwargs
    )


def test_original_receipt_bytes_hmac_and_fresh_nonce_with_exact_server_ack(scene):
    terminal(scene, success=True)
    item = outbox(scene).claim_due(NOW)
    attestations = []
    sender, transport = client(item, before_mutation=lambda: attestations.append("attest"))
    for nonce in ("fresh-nonce-0001", "fresh-nonce-0002"):
        result = sender.run_upload_receipt_once(item)
        assert isinstance(result, SyncAttemptResult) and result.outcome == "complete"
        assert result.receipt_ack == ack(item)
        path, headers, body = transport.calls[-1]
        assert path == "/v1/sync/saved-clips/cloud-job-1/upload-receipt"
        assert body == item.receipt_json.encode("ascii")
        assert headers["x-nonce"] == nonce
        assert headers["x-device-id"] == SCOPE.device_id
        digest = hashlib.sha256(body).hexdigest()
        assert headers["x-body-sha256"] == digest
        canonical = f"POST\n{path}\n1793491200\n{nonce}\n{digest}".encode()
        assert (
            headers["x-signature"]
            == hmac.new(b"synthetic-secret", canonical, hashlib.sha256).hexdigest()
        )
    assert attestations == ["attest", "attest"]
    assert set(SyncAttemptResult.__dataclass_fields__) == {"outcome", "error_code"}


@pytest.mark.parametrize("change", ["extra", "request", "hash", "false", "date", "precision"])
def test_client_reuses_strict_queue_ack_validation(scene, change):
    terminal(scene)
    item = outbox(scene).claim_due(NOW)
    value = ack(item)
    if change == "extra":
        value["extra"] = None
    if change == "request":
        value["upload_request"]["request_id"] = "wrong"
    if change == "hash":
        value["receipt_sha256"] = "0" * 64
    if change == "false":
        value["erasable"] = 0
    if change == "date":
        value["received_at_utc"] = "2026-02-30T00:00:00.000Z"
    if change == "precision":
        value["received_at_utc"] = "2026-11-01T00:00:00.000000Z"
    sender, _ = client(item, HttpResult(200, json.dumps({"data": value}).encode()))
    result = sender.run_upload_receipt_once(item)
    assert (result.outcome, result.error_code, result.receipt_ack) == (
        "blocked",
        "receipt_invalid",
        None,
    )


@pytest.mark.parametrize(
    "status,code,outcome,safe",
    [
        (409, "SYNC_CONFLICT", "blocked", "receipt_conflict"),
        (400, "SYNC_INVALID", "blocked", "receipt_invalid"),
        (503, "SYNC_STORAGE_UNAVAILABLE", "retry", "receipt_unavailable"),
        (401, "SYNC_UNAUTHORIZED", "retry", "receipt_auth_failed"),
        (403, "private https://secret", "retry", "receipt_auth_failed"),
        (500, "private https://secret", "retry", "receipt_unavailable"),
    ],
)
def test_fixed_error_classification_auth_stays_pending(scene, status, code, outcome, safe):
    terminal(scene)
    item = outbox(scene).claim_due(NOW)
    sender, _ = client(item, HttpResult(status, json.dumps({"error": {"code": code}}).encode()))
    result = sender.run_upload_receipt_once(item)
    assert (result.outcome, result.error_code, result.receipt_ack) == (outcome, safe, None)


@pytest.mark.parametrize("failure", [OSError("private"), ValueError("attestation private")])
def test_attestation_unavailable_never_sends_or_drops_body(scene, failure):
    terminal(scene)
    item = outbox(scene).claim_due(NOW)

    def unavailable():
        raise failure

    sender, transport = client(item, before_mutation=unavailable)
    result = sender.run_upload_receipt_once(item)
    assert result.outcome == "retry" and result.receipt_ack is None
    assert transport.calls == []
    assert rows(scene[0])[0]["receipt_json"] == item.receipt_json


@pytest.mark.parametrize("saved_status", ["acknowledged", "blocked", "owner_deleted"])
def test_actual_worker_delivers_without_saved_claim_file_or_historical_model(
    scene, saved_status, monkeypatch
):
    from tests.unit.edge.route_assistance.test_owner_media_deletion_authority import (
        owner_module,
        payload,
    )

    terminal(scene, success=True)
    store, saved = scene
    if saved_status == "owner_deleted":
        store.accept_owner_delete_task(
            owner_module().OwnerDeleteDeviceTask.from_payload(payload(store, shot="shot-0002")),
            managed_scope=SCOPE,
        )
    else:
        with sqlite3.connect(store.path) as db:
            db.execute(
                "UPDATE field_saved_media_outbox SET status=?,claim_token=NULL,"
                "lease_until_utc=NULL",
                (saved_status,),
            )
    saved.local_path.unlink()
    original = store.get_managed_resource_spec_for_session(SCOPE, saved.session_id)
    assert not Path(original.spec.paths["detector_training_run"]).exists()
    from tests.unit.edge.route_assistance.test_managed_session_persistence import snapshot

    before = snapshot(store.path)

    def forbidden(*args, **kwargs):
        pytest.fail("receipt accessed media or current saved authority")

    monkeypatch.setattr(store, "claim_due_saved_media", forbidden)
    monkeypatch.setattr(store, "is_saved_media_claim_current", forbidden)
    monkeypatch.setattr(
        "billiards_ai.lab.field_assistance.cloud_runtime.session_media_lock", forbidden
    )

    # This synthetic fixture intentionally retains original metadata only.
    class AckTransport(Transport):
        def post_json(self, path, headers, body):
            self.calls.append((path, headers, body))
            receipt = json.loads(body)
            value = dict(
                contract="managed_upload_receipt_ack_v1",
                upload_request=receipt["upload_request"],
                receipt_sha256=hashlib.sha256(body).hexdigest(),
                received_at_utc="2026-11-01T00:00:00.123Z",
                erasable=False,
            )
            return HttpResult(200, json.dumps({"data": value}).encode())

    transport = AckTransport(None)
    sender = FieldSyncClient(credentials=Credentials(), transport=transport)
    drain = worker(store, sender)
    assert drain._join_timeout_seconds == 45
    assert drain.run_once() is True
    assert rows(store)[0]["status"] == "acknowledged"
    assert len(transport.calls) == 1 and drain.run_once() is False
    after = snapshot(store.path)
    assert {k: v for k, v in before.items() if k != "field_upload_receipt_outbox"} == {
        k: v for k, v in after.items() if k != "field_upload_receipt_outbox"
    }


def test_response_loss_retry_exact_replay_and_stale_ack_cannot_settle(scene):
    terminal(scene)
    store = scene[0]
    first = outbox(scene).claim_due(NOW, lease_seconds=1)
    sender, transport = client(first, OSError("lost after server commit"))
    clock = [NOW + timedelta(seconds=2)]
    drain = worker(store, sender)
    drain._now_utc = lambda: clock[0]
    assert drain.run_once() is True
    pending = rows(store)[0]
    assert pending["status"] == "pending" and pending["last_error_code"] == "receipt_unavailable"
    assert outbox(scene).acknowledge(first, ack(first)) is False
    assert drain.run_once() is False
    transport.response = HttpResult(200, json.dumps({"data": ack(first)}).encode())
    clock[0] += timedelta(seconds=2)
    assert drain.run_once() is True
    assert rows(store)[0]["status"] == "acknowledged"
    assert transport.calls[0][2] == transport.calls[1][2] == first.receipt_json.encode()
    assert transport.calls[0][1]["x-nonce"] != transport.calls[1][1]["x-nonce"]


def test_corrupt_queue_is_not_sent_and_foreign_scope_does_not_claim(scene):
    terminal(scene)
    sender = SimpleNamespace(run_upload_receipt_once=lambda item: pytest.fail("invalid claim sent"))
    drain = worker(scene[0], sender)
    drain._outbox = outbox(scene, replace(SCOPE, device_id="foreign"))
    assert drain.run_once() is False
    with sqlite3.connect(scene[0].path) as db:
        db.execute("UPDATE field_upload_receipt_outbox SET receipt_json='{}'")
    assert worker(scene[0], sender).run_once() is False
    assert rows(scene[0])[0]["last_error_code"] == "receipt_invalid"


def test_worker_requires_scope_and_cancellation_leaves_owned_lease_recoverable(scene):
    terminal(scene)
    module = importlib.import_module("billiards_ai.lab.field_assistance.cloud_runtime")
    with pytest.raises(ValueError):
        module._BackgroundUploadReceiptSyncWorker(
            store=scene[0], client=object(), managed_scope=None
        )
    sender = SimpleNamespace()
    drain = worker(scene[0], sender)

    def cancel(item):
        drain.stop()
        return importlib.import_module(
            "billiards_ai.edge.route_assistance.sync"
        ).UploadReceiptSyncResult("complete", receipt_ack=ack(item))

    sender.run_upload_receipt_once = cancel
    assert drain.run_once() is True
    assert rows(scene[0])[0]["status"] == "leased"
    assert outbox(scene).claim_due(NOW + timedelta(seconds=61)) is not None


def test_worker_reconciles_sixteen_and_sends_one_with_opaque_cursor(scene):
    records = [terminal(scene) for _ in range(17)]
    with sqlite3.connect(scene[0].path) as db:
        db.execute("DELETE FROM field_upload_receipt_outbox")
    original_ids = sorted(record.request_id for record in records)

    # A real client accepts only the ACK for the exact original request just sent.
    class Reply:
        calls = []

        def post_json(self, path, headers, body):
            self.calls.append(body)
            item = json.loads(body)
            value = dict(
                contract="managed_upload_receipt_ack_v1",
                upload_request=item["upload_request"],
                receipt_sha256=hashlib.sha256(body).hexdigest(),
                received_at_utc="2026-11-01T00:00:00.001Z",
                erasable=False,
            )
            return HttpResult(200, json.dumps({"data": value}).encode())

    transport = Reply()
    drain = worker(scene[0], FieldSyncClient(credentials=Credentials(), transport=transport))
    assert drain.run_once() is True
    assert len(rows(scene[0])) == 16 and drain._cursor == original_ids[15]
    assert len(transport.calls) == 1
    assert drain.run_once() is True
    assert len(rows(scene[0])) == 17 and drain._cursor == original_ids[16]
    assert len(transport.calls) == 2
    assert drain.run_once() is True
    assert drain._cursor is None and len(transport.calls) == 3


@pytest.mark.parametrize(
    "status,code,state,safe",
    [
        (409, "SYNC_CONFLICT", "blocked", "receipt_conflict"),
        (400, "SYNC_INVALID", "blocked", "receipt_invalid"),
        (401, "SYNC_UNAUTHORIZED", "pending", "receipt_auth_failed"),
    ],
)
def test_actual_worker_fixed_error_settlement(scene, status, code, state, safe):
    terminal(scene)
    item = outbox(scene).claim_due(NOW, lease_seconds=1)
    sender, _ = client(item, HttpResult(status, json.dumps({"error": {"code": code}}).encode()))
    drain = worker(scene[0], sender)
    drain._now_utc = lambda: NOW + timedelta(seconds=2)
    assert drain.run_once() is True
    row = rows(scene[0])[0]
    assert row["status"] == state and row["last_error_code"] == safe
    if state == "pending":
        assert row["next_attempt_at_utc"] == "2026-11-01T00:00:04.000000Z"
        assert drain.run_once() is False


def test_worker_propagates_exact_ack_object_and_ignores_stale_settlement(scene, monkeypatch):
    terminal(scene)
    item = outbox(scene).claim_due(NOW, lease_seconds=1)
    sender, _ = client(item)
    drain = worker(scene[0], sender)
    drain._now_utc = lambda: NOW + timedelta(seconds=2)
    actual = sender.run_upload_receipt_once
    result_ack = []

    def send(item):
        result = actual(item)
        result_ack.append(result.receipt_ack)
        # Another owner reclaims before the original response is settled.
        assert outbox(scene).claim_due(NOW + timedelta(seconds=63)) is not None
        return result

    sender.run_upload_receipt_once = send
    acknowledged = []
    settle = drain._outbox.acknowledge

    def acknowledge(item, data):
        acknowledged.append(data)
        return settle(item, data)

    monkeypatch.setattr(drain._outbox, "acknowledge", acknowledge)
    assert drain.run_once() is True
    assert acknowledged[0] is result_ack[0]
    assert rows(scene[0])[0]["status"] == "leased"


def test_only_metadata_control_pin_is_acquired_and_never_media_pin(scene, monkeypatch):
    from contextlib import contextmanager

    from billiards_ai.edge.route_assistance import saved_upload_requests

    terminal(scene)
    item = outbox(scene).claim_due(NOW, lease_seconds=1)
    sender, _ = client(item)
    drain = worker(scene[0], sender)
    drain._now_utc = lambda: NOW + timedelta(seconds=2)
    original = saved_upload_requests.session_media_lock
    paths = []

    @contextmanager
    def metadata_pin(path, *args, **kwargs):
        paths.append(path)
        assert path.name == ".saved-upload-request-control"
        with original(path, *args, **kwargs):
            yield

    monkeypatch.setattr(saved_upload_requests, "session_media_lock", metadata_pin)
    assert drain.run_once() is True
    assert paths and rows(scene[0])[0]["status"] == "acknowledged"


def test_receipt_thread_stop_timeout_keeps_lease_then_no_late_ack(scene):
    import threading

    terminal(scene)
    item = outbox(scene).claim_due(NOW, lease_seconds=1)
    sender, _ = client(item)
    entered, release = threading.Event(), threading.Event()
    actual = sender.run_upload_receipt_once

    def send(item):
        entered.set()
        assert release.wait(2)
        return actual(item)

    sender.run_upload_receipt_once = send
    drain = worker(scene[0], sender, join_timeout_seconds=0.01, poll_seconds=0.001)
    drain._now_utc = lambda: NOW + timedelta(seconds=2)
    drain.start()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="receipt sync worker"):
            drain.stop()
        assert rows(scene[0])[0]["status"] == "leased"
    finally:
        release.set()
        drain._thread.join(2)
        drain.stop()
    assert not drain._thread.is_alive()
    assert rows(scene[0])[0]["status"] == "leased"


def test_mismatched_current_device_never_sends_original_scope_receipt(scene):
    terminal(scene)
    item = outbox(scene).claim_due(NOW)
    sender, transport = client(item)
    sender._credentials = SimpleNamespace(device_id=lambda: "reassigned-device")
    result = sender.run_upload_receipt_once(item)
    assert (result.outcome, result.error_code, result.receipt_ack) == (
        "retry",
        "receipt_auth_failed",
        None,
    )
    assert transport.calls == []


def test_worker_retry_backoff_is_bounded_and_receipt_bytes_never_change(scene):
    terminal(scene)
    item = outbox(scene).claim_due(NOW, lease_seconds=1)
    transport = Transport(HttpResult(401, b'{"error":{"code":"SYNC_UNAUTHORIZED"}}'))
    sender = FieldSyncClient(credentials=Credentials(), transport=transport)
    clock = [NOW + timedelta(seconds=2)]
    drain = worker(scene[0], sender)
    drain._now_utc = lambda: clock[0]
    delays = []
    for _ in range(12):
        assert drain.run_once() is True
        row = rows(scene[0])[0]
        due = datetime.fromisoformat(row["next_attempt_at_utc"])
        delays.append((due - clock[0]).total_seconds())
        assert drain.run_once() is False
        clock[0] = due
    assert delays[0] == 2 and delays[-1] == max(delays) == 300
    assert {body for _, _, body in transport.calls} == {item.receipt_json.encode()}
